import {processProduct,autocomplete,normalize,stockPolicy,stockLast} from './core/core.mjs';
import {applyRanking,wordMatches} from './core/ranking.mjs';
import {orderSeries} from './core/series.mjs';
import {createSearchService} from './core/semantic.mjs';
import {generate} from './core/gemini.mjs';
import {processGarmin} from './tenants/garmin/index.mjs';
import {hash} from './core/hash.mjs';
import {createIndexRetriever,buildSearchIndex,TOKENIZER} from './core/search-index.mjs';
import {createVectorRanker,createPackedRanker} from './core/embeddings.mjs';
import {createHookRunner,hasHooks} from './core/tenant-hooks.mjs';
export function createDraftRuntime(project,revision,{retrieve:overrideRetrieve,releaseBase=false}={}) {
 const profile={...revision.profile,storeContext:project.storeContext,tenantId:project.id,platform:project.platform,sourceUrl:project.url,version:`studio-${revision.number}`,publishedStatuses:['ACTIVE','publish'],indexPlan:{text:['title','description','specifications'],filter:['productType','colors','finishes','tags','price','stockStatus'],exact:['id','sku','mpn','gtin']}};
 const classifiedTags=new Map();
 for(const [tag,assignment] of Object.entries(project.tagAssignments||{})) {
  if(!Object.hasOwn(profile.tagDefinitions||{},tag)||assignment.definitionHash&&assignment.definitionHash!==hash(profile.tagDefinitions[tag]))continue;
  for(const id of assignment.matchedIds||[]){if(!classifiedTags.has(id))classifiedTags.set(id,[]);classifiedTags.get(id).push(tag)}
 }
 let products=project.productCards?.length&&project.productCardsProfileHash===hash(revision.profile)?project.productCards:project.catalog.products.map(raw=>{
  if(new URL(project.url).hostname.replace(/^www\./,'')==='garmin.co.il')raw=processGarmin(raw);
  raw={...raw,tags:[...new Set([...(raw.tags||[]),...(raw.siteTags||[]),...(classifiedTags.get(String(raw.id))||[])])]};
  const p=processProduct(raw,profile);p.sku=raw.sku||p.sku;if(raw.popularity)p.popularity=raw.popularity;p.mpn=raw.mpn;p.gtin=raw.gtin;p.variants=raw.variants||[];p.priceRange=raw.priceRange;p.badges=Array.isArray(raw.badges)?raw.badges.filter(b=>{if(b.kind==='sale'&&!(Number.isFinite(raw.regularPrice)&&Number.isFinite(raw.price)&&raw.regularPrice>raw.price)){p.issues.push('source-sale-conflict');return false;}if(b.kind==='stock'&&raw.stockStatus!=='outofstock'){p.issues.push('source-stock-conflict');return false;}return true;}).map(b=>({...b})):[];
  if(raw.garmin){p.finishes=raw.garmin.finishes||[];p.garmin=raw.garmin;if(raw.garmin.accessory)p.productType=profile.productTypes.band?.categories.some(c=>p.categories.includes(c))?'band':null;}
  for(const r of profile.badgeRules)if((raw[r.field]||[]).includes(r.value)&&!p.badges.some(b=>b.text===r.text))p.badges.push({text:r.text,kind:'merchant',order:r.order,source:'tenant-rule'});
  p.badges.sort((a,b)=>(a.order||0)-(b.order||0));return p;
 });
 // WooCommerce's popular products (tag isWooPopular, from the store's own flag) are lifted by default, among equally
 // relevant results, unless the tenant already ranks by that tag.
 if(products.some(p=>p.tags?.includes('isWooPopular'))&&!(profile.rankingRules||[]).some(r=>r.field==='tags'&&r.values?.includes('isWooPopular')))
  profile.rankingRules=[...(profile.rankingRules||[]),{id:'rank-woo-popular',name:'מוצרים פופולריים (isWooPopular) קודם',action:'boost',field:'tags',values:['isWooPopular'],terms:[],match:'direct'}];
 const contentHash=hash(products);
 let index=project.searchIndex?.tokenizer===TOKENIZER&&project.searchIndex?.contentHash===contentHash?project.searchIndex:buildSearchIndex(products,profile.version);
 let retrieve=overrideRetrieve||createIndexRetriever(products,profile,index);
 // A build's vector index (exact content) wins; otherwise the studio's per-product vectors from the processing lab.
 const rankCandidates=project.vectorIndex?.contentHash===contentHash?createVectorRanker(project.vectorIndex):project.studioVectors?.ids?.length?createPackedRanker(project.studioVectors):null;
 // Tenant functions: rewriteQuery/rerank run per search; transformProduct reshapes the catalog once (in its isolated
 // worker) and the index is rebuilt on the result. If the transform fails, the untransformed catalog keeps serving.
 let hooks=null,hookError=null;if(hasHooks(profile))try{hooks=createHookRunner(profile.hooks,profile.hookData,{tenant:project.id});}catch(e){hookError=e.message;}
 const service=(list,listRetrieve=retrieve)=>createSearchService(list,profile,generate,{...profile.pipeline,maxEntries:10,retrieve:listRetrieve,rankCandidates,hooks});
 let search=service(products);
 let suggestIndex=index,suggestRetrieve=retrieve,prepareProducts=null;
 if(hooks?.has('transformProduct')){let base=search;let ready=null;
  const prepare=()=>ready??=hooks.transformProducts(products).then(list=>{
   suggestIndex=buildSearchIndex(list,profile.version);
   suggestRetrieve=overrideRetrieve||createIndexRetriever(list,profile,suggestIndex);
   // A long-lived module keeps one catalog: holding the untransformed one too cost Shipi 216 MB of a 1 GB heap (Nov 2026).
   if(releaseBase){products=list;index=suggestIndex;retrieve=suggestRetrieve;base=null;project.productCards=null;}
   return service(list,suggestRetrieve);
  }).catch(e=>{hookError='transformProduct: '+e.message;return base;});
  prepareProducts=prepare;
  search=async request=>(await prepare())(request);search.ready=prepare;}
 // Suggestions while the shopper types. The same rules as search decide (the tenant's rewriteQuery function, spelling
 // and linked terms, tag / type / price filters, hidden products and the stock policy, the index's order) — but only
 // from the index: no model call, so it answers in the time of a keystroke. The last word may be unfinished, so the
 // catalog's own words that begin with it are tried too, most common first. Each list is ordered the way search orders
 // it — the tenant's ranking rules (boost / bury) for the words it was found with — then products whose title carries
 // the typed words come before products that only mention them elsewhere, out-of-stock products go last under that
 // policy, and the tenant's rerank function (if any) has the last word.
 async function suggest(query,{limit=8}={}){
  if(!Number.isInteger(limit)||limit<1||limit>50)throw Error('Invalid limit');
  let q=typeof query==='string'?query.trim().slice(0,300):'';if(normalize(q).length<2)return {query:q,matches:[],total:0};
  if(prepareProducts)await prepareProducts();
  if(hooks?.has('rewriteQuery'))try{q=String(await hooks.rewriteQuery(q)||q);}catch{}
  // One row per title: catalogs carry the same product under more than one id.
  const seen=new Set(),titles=new Set(),matches=[],applied=new Set();
  const take=(r,words)=>{const ranked=applyRanking(r?.matches||[],profile.rankingRules,words);for(const name of ranked.applied)applied.add(name);
   for(const p of ranked.matches){const t=normalize(p.title);if(seen.has(p.id)||titles.has(t))continue;seen.add(p.id);titles.add(t);matches.push(p);}};
  const literal=await suggestRetrieve(q);take(literal,q);
  const words=q.split(/\s+/),last=normalize(words.at(-1)),completed=[];
  if(matches.length<limit&&last.length>=2&&!last.includes(' ')){
   const terms=Object.keys(suggestIndex.terms||{}).filter(t=>t!==last&&t.startsWith(last)).sort((a,b)=>suggestIndex.terms[b].length-suggestIndex.terms[a].length||a.localeCompare(b)).slice(0,6);
   for(const t of terms){if(matches.length>=limit*3)break;const before=matches.length;const full=[...words.slice(0,-1),t].join(' ');take(await suggestRetrieve(full),full);if(matches.length>before)completed.push(t);}
  }
  const typed=normalize(q).split(' ').filter(Boolean),inTitle=p=>{const title=normalize(p.title).split(' ');return typed.every(w=>title.some(t=>t.startsWith(w))||wordMatches(w,title))?1:0;};
  let ranked=matches.map((p,i)=>({p,i,t:inTitle(p)})).sort((a,b)=>b.t-a.t||a.i-b.i).map(x=>x.p);
  const functions=[];if(hooks?.has('rerank')&&ranked.length)try{ranked=await hooks.rerank(ranked.slice(0,Math.max(limit*3,60)),q);functions.push('rerank');}catch{}
  // Same as search: the series order, then out-of-stock last, both after the tenant's function.
  if(profile.pipeline?.seriesOrder)ranked=orderSeries(ranked,q,{scoped:literal?.plan?.scopedAliases?.some(r=>r.mode==='only')});
  if(!ranked.length&&(profile.hookData?.shipiSearchRules?.alwaysResults===true||profile.hookData?.catalogSearchRules?.alwaysResults===true)){
   const candidates=(await suggestRetrieve('')).matches||[],terms=normalize(query).split(' ').filter(t=>t.length>1);
   ranked=candidates.map((p,i)=>({p,i,s:terms.reduce((s,t)=>s+(normalize(p.title).includes(t)?1:0),0)})).sort((a,b)=>b.s-a.s||a.i-b.i).slice(0,Math.max(limit*3,60)).map(x=>x.p);
   if(hooks?.has('rerank')&&ranked.length)try{ranked=await hooks.rerank(ranked,q);}catch{}
   if(!ranked.length)ranked=candidates.slice(0,limit);
   ranked=ranked.map(p=>({...p,matchQuality:'alternative',alternativeReason:'הצעה קרובה מהקטלוג; יש לבדוק את הדגם והמאפיינים בכותרת.'}));
  }
  if((profile.hookData?.shipiSearchRules?.alwaysResults===true||profile.hookData?.catalogSearchRules?.alwaysResults===true)){
   const required=q.match(/\bsxguard0[a-z0-9]+\b/g)||[],forbidden=(q.match(/\bsxavoid0[a-z0-9]+\b/g)||[]).map(t=>t.replace(/^sxavoid0/,'sxguard0'));
   ranked=ranked.map(p=>required.some(t=>!(p.tags||[]).includes(t))||forbidden.some(t=>(p.tags||[]).includes(t))?{...p,matchQuality:'alternative',alternativeReason:'הצעה קרובה מהקטלוג; יש לבדוק את הדגם והמאפיינים בכותרת.'}:p);
  }
  if(stockPolicy(profile)==='last')ranked=stockLast(ranked,q);
  return {query:q,matches:ranked.slice(0,limit),total:ranked.length,plan:literal?.plan||null,completed,ranking:[...applied],functions};
 }
 return {search,suggest,autocomplete:query=>autocomplete(products,profile,query),get products(){return products;},profile,get index(){return index;},functions:hooks?.names||[],get functionError(){return hookError;}};
}
export function indexDefinition(profile={}) {
 const fields={name:[{type:'string'},{type:'autocomplete'}],id:{type:'token'},sku:{type:'token'},mpn:{type:'token'},gtin:{type:'token'},variants:{type:'document',dynamic:false,fields:{id:{type:'token'},sku:{type:'token'},mpn:{type:'token'},gtin:{type:'token'}}},description:{type:'string'},specifications:{type:'document',dynamic:true},tenantId:{type:'token'},categories:[{type:'token'},{type:'string'}],collections:[{type:'token'},{type:'string'}],tags:{type:'token'},colors:{type:'token'},finishes:{type:'token'},productType:{type:'token'},price:{type:'number'},stockStatus:{type:'token'},hidden:{type:'boolean'}};
 const selected=[...new Set([...Object.keys(fields),...(profile.indexFields||[])])];
 return {name:'tenant_products_v1',definition:{mappings:{dynamic:false,fields:Object.fromEntries(selected.map(k=>[k,fields[k]]))}}};
}
