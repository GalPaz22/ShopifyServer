import {normalize,planQuery,matchesScopedAliases,sellable,authorMatch} from './core.mjs';
import {hash} from './hash.mjs';
// Bump when tokenization changes so a saved index is rebuilt instead of reused.
export const TOKENIZER=3;
export function buildSearchIndex(products,version) {
  const terms=Object.create(null),identifiers=Object.create(null);
  for(const p of products){const values=[p.id,p.sku,p.mpn,p.gtin,...(p.variants||[]).filter(v=>v.stockStatus!=='outofstock').flatMap(v=>[v.id,v.sku,v.mpn,v.gtin])];for(const k of new Set(values.filter(Boolean).map(normalize)))(identifiers[k]??=[]).push(p.id);
    const text=normalize([p.title,p.description,p.summary,...p.categories,...p.tags,...Object.values(p.specifications||{})].join(' '));
    for(const token of new Set(text.split(' ').filter(Boolean)))(terms[token]??=[]).push(p.id);
  }
  return {version,kind:'local-inverted',tokenizer:TOKENIZER,documents:products.length,contentHash:hash(products),terms,identifiers,builtAt:new Date().toISOString()};
}
// Returns the edit distance, or max+1 as soon as it is certain to exceed max.
export function boundedDistance(a,b,max){
  if(Math.abs(a.length-b.length)>max)return max+1;
  let row=Array.from({length:b.length+1},(_,i)=>i);
  for(let i=1;i<=a.length;i++){const next=[i];let best=i;for(let j=1;j<=b.length;j++){next[j]=Math.min(next[j-1]+1,row[j]+1,row[j-1]+(a[i-1]===b[j-1]?0:1));if(next[j]<best)best=next[j];}if(best>max)return max+1;row=next;}
  return row[b.length];
}
const hebrew=/^[\u05d0-\u05ea]+$/u,prefixes=/^[והבלמשכ]/u,ATTACHED=['ה','ו','ב','ל','וה','וב','ול'];
const vocabularies=new WeakMap();
// Hebrew number inflection: the singular and plural of a word are the same product word (בננות↔בננה,
// עגבניות↔עגבניה/עגבנייה, תפוחים↔תפוח, מלפפונים↔מלפפון). Only forms that exist in the index are used.
const FINAL={ך:'כ',ם:'מ',ן:'נ',ף:'פ',ץ:'צ'},TO_FINAL=Object.fromEntries(Object.entries(FINAL).map(([f,r])=>[r,f]));
const open=w=>w.slice(0,-1)+(FINAL[w.at(-1)]||w.at(-1)),close=w=>w.slice(0,-1)+(TO_FINAL[w.at(-1)]||w.at(-1));
export function inflections(term){
  if(!hebrew.test(term)||term.length<3)return [];const out=new Set();const stem=(w,n)=>{const b=w.slice(0,-n);return b.length>=3?b:null;};
  if(term.endsWith('יות')){const b=stem(term,3);if(b)for(const x of [b+'יה',b+'ייה',b+'ית'])out.add(x);}
  if(term.endsWith('ות')){const b=stem(term,2);if(b)for(const x of [b+'ה',b+'ת',close(b)])out.add(x);}
  if(term.endsWith('ים')){const b=stem(term,2);if(b)out.add(close(b));}
  if(term.endsWith('ה')&&term.length>=4){const b=term.slice(0,-1);out.add(b+'ות');if(b.endsWith('י'))out.add(b.slice(0,-1)+'יות');}
  if(!/(ות|ים|ה)$/.test(term)){const b=open(term);out.add(b+'ים');out.add(b+'ות');}
  out.delete(term);return [...out];
}
function vocabulary(index){
  if(!vocabularies.has(index)){const byLength=new Map();for(const t of Object.keys(index.terms))if(!/\d/.test(t))(byLength.get(t.length)||byLength.set(t.length,[]).get(t.length)).push(t);vocabularies.set(index,byLength);}
  return vocabularies.get(index);
}
// Exact first; for a term without any posting: Hebrew prefix stripping, then bounded fuzzy.
// relax=true also approximates a term that exists, keeping its own postings (a rare real word
// such as "שליו" must not block the correction to "שלו").
// Terms with digits (ISBN, models) and short words are never approximated.
// partial=true (a tenant's pipeline.partialWords) also reads an unfinished word as the catalog words it begins
// ("ארנ" → ארנק, "סמסו" → סמסונג), most common first — after the exact word and its inflections, beside the fuzzy ones.
export function resolveTerm(index,term,{relax=false,partial=false}={}){
  const own=new Map((Object.hasOwn(index.terms,term)?index.terms[term]:[]).map(id=>[id,0]));
  // The other number of the same word counts as the word itself (ranked just after the exact form).
  const forms=inflections(term).filter(t=>Object.hasOwn(index.terms,t));for(const t of forms)for(const id of index.terms[t])if(!own.has(id))own.set(id,0.5);
  // The same word behind a Hebrew prefix is the word itself when the catalog mostly writes it that way: categories say
  // "שעונים לנשים", shoppers type "נשים" (which only one product spelled bare). Used only when the prefixed forms cover
  // more products than the bare word does, so common words ("אדם" / "האדם" in blurbs) keep their precise results.
  // Only ה/ו/ב/ל (and ו before them): מ/ש/כ are too often the word's own first letter (מספר, שמן).
  const prefixed=[],extra=new Set();if(hebrew.test(term)&&term.length>=3)for(const base of [term,...forms])for(const p of ATTACHED){const t=p+base;if(Object.hasOwn(index.terms,t)){prefixed.push(t);for(const id of index.terms[t])if(!own.has(id))extra.add(id);}}
  // A word with no bare form at all keeps the spelling and typo fallbacks below.
  if(own.size&&extra.size>own.size)for(const id of extra)own.set(id,0.5);else prefixed.length=0;
  if(own.size&&!relax)return {kind:Object.hasOwn(index.terms,term)?'exact':forms.length?'inflection':'prefixed',...(forms.length&&{inflections:forms}),...(prefixed.length&&{prefixed}),postings:own};
  // A leading ו/ה/ב/ל/מ/ש/כ may be a prefix or part of the word (שליו), so both readings are kept.
  let stem=null;if(hebrew.test(term))for(let s=term,i=0;i<3&&prefixes.test(s)&&s.length>3;i++){s=s.slice(1);if(Object.hasOwn(index.terms,s)){stem=s;break;}}
  const found=[];
  if(term.length>=4&&!/\d/.test(term)){const max=term.length>=8?2:1,byLength=vocabulary(index);
    for(let len=term.length-max;len<=term.length+max;len++)for(const t of byLength.get(len)||[]){if(t===term)continue;const edits=boundedDistance(term,t,max);if(edits<=max)found.push({t,edits});}}
  const least=found.length?Math.min(...found.map(f=>f.edits)):0,chosen=found.filter(f=>f.edits===least).sort((a,b)=>index.terms[b.t].length-index.terms[a.t].length).slice(0,10);
  const completions=partial&&!own.size&&term.length>=3&&!/^\d+$/.test(term)?Object.keys(index.terms).filter(t=>t.length>term.length&&t.startsWith(term)).sort((a,b)=>index.terms[b].length-index.terms[a].length).slice(0,20):[];
  if(!stem&&!chosen.length&&!completions.length)return {kind:'none',postings:own};
  const postings=new Map(own);const add=(t,cost)=>{for(const id of index.terms[t])if(!postings.has(id)||postings.get(id)>cost)postings.set(id,cost);};
  if(stem)add(stem,1);for(const t of completions)add(t,1);for(const {t,edits} of chosen)add(t,1+edits);
  if(completions.length&&!stem&&!chosen.length)return {kind:'completion',to:completions,postings};
  // relaxed: the term is a real catalog word that was approximated only because the words do not occur together.
  return {kind:stem&&!chosen.length?'prefix':chosen.length&&!stem?'fuzzy':'prefix+fuzzy',to:[...(stem?[stem]:[]),...chosen.map(c=>c.t)],...(chosen.length&&{edits:least}),...(relax&&own.size&&{relaxed:true}),postings};
}
export function createIndexRetriever(products,client,index) {
  if(index.contentHash!==hash(products))throw Error('האינדקס אינו תואם לגרסת המוצרים');
  const byId=new Map(products.map(p=>[p.id,p]));
  return (query)=>{
    const partial=client.pipeline?.partialWords===true,normalized=normalize(query),plan=planQuery(query,client),exact=Object.hasOwn(index.identifiers,normalized)?index.identifiers[normalized]:null;
    // An exact identifier (SKU, barcode, id) finds the product even when it is out of stock; hidden ones stay hidden.
    const visible=p=>exact?!p.hidden:sellable(p,client)&&(
      matchesScopedAliases(p,plan)&&(!plan.productType||p.productType===plan.productType)&&plan.colors.every(c=>p.colors.includes(c))&&plan.finishes.every(f=>p.finishes.includes(f))&&plan.tags.every(t=>p.tags.includes(t))&&(plan.maxPrice===null||p.price!==null&&p.price<=plan.maxPrice));
    const run=resolved=>{let ids=exact||products.map(p=>p.id);const penalty=new Map();
      if(!exact)for(const r of resolved){ids=ids.filter(id=>r.postings.has(id));for(const id of ids)penalty.set(id,(penalty.get(id)||0)+r.postings.get(id));}
      return {penalty,matches:ids.map(id=>byId.get(id)).filter(visible),resolved};};
    let resolved=exact?[]:plan.terms.map(term=>resolveTerm(index,term,{partial})),best=run(resolved);
    // Every term exists but not together: approximate one term at a time, fewest edits first.
    if(!best.matches.length&&resolved.length>1)for(const [i,term] of plan.terms.entries()){if(resolved[i].kind!=='exact')continue;const relaxed=resolveTerm(index,term,{relax:true,partial});if(relaxed.kind==='none')continue;
      const attempt=run(resolved.map((r,j)=>j===i?relaxed:r));if(attempt.matches.length&&(!best.matches.length||(relaxed.edits||0)<(best.relaxedEdits||0)||(relaxed.edits||0)===(best.relaxedEdits||0)&&attempt.matches.length>best.matches.length))best={...attempt,relaxedEdits:relaxed.edits||0};}
    const {penalty}=best;let {matches}=best;
    const corrections=best.resolved.flatMap((r,i)=>r.kind!=='exact'&&r.kind!=='none'?[{term:plan.terms[i],kind:r.kind,to:r.to,...(r.edits&&{edits:r.edits}),...(r.relaxed&&{relaxed:true})}]:[]);
    if(corrections.length)plan.corrections=corrections;
    const rankTerms=plan.terms.length?plan.terms:(plan.scopedAliases||[]).map(r=>normalize(r.term)).filter(Boolean);
    const titleHits=p=>{const words=new Set(normalize(p.title).split(' ').filter(Boolean));return rankTerms.reduce((n,t)=>n+(words.has(t)?1:0),0);};
    // A query that names an author puts that author's books first ("חיים נבון" over blurbs that mention both words).
    // Among equally good matches, products shoppers chose more often (production clicks and carts) come first.
    const byAuthor=new Set(rankTerms.length?matches.filter(p=>authorMatch(p,rankTerms.join(' '))).map(p=>p.id):[]);
    matches.sort((a,b)=>(penalty.get(a.id)||0)-(penalty.get(b.id)||0)||Number(byAuthor.has(b.id))-Number(byAuthor.has(a.id))||titleHits(b)-titleHits(a)||Number(normalize(b.title)===normalize(query))-Number(normalize(a.title)===normalize(query))||(b.popularity||0)-(a.popularity||0)||a.id.localeCompare(b.id));
    if(exact)matches=matches.map(p=>{const variant=(p.variants||[]).find(v=>[v.id,v.sku,v.mpn,v.gtin].filter(Boolean).some(id=>normalize(id)===normalized));return variant?{...p,matchedVariant:variant}:p;});
    if(!exact){plan.termHits=plan.terms.map((term,i)=>{const r=best.resolved[i],exactCount=index.terms[term]?.length||0;return {term,exact:exactCount,kind:r?.kind||'none',to:r?.to,postings:r?.postings.size??0};});
      if(!matches.length)plan.why=plan.scopedProductIds&&!plan.terms.length?{kind:'scoped-only',term:plan.scopedAliases[0]?.term,products:plan.scopedProductIds.length}:plan.termHits.some(t=>!t.postings)?{kind:'missing-terms',terms:plan.termHits.filter(t=>!t.postings).map(t=>t.term)}:plan.termHits.length>1&&plan.termHits.every(t=>t.exact)?{kind:'no-intersection',terms:plan.termHits.map(t=>`${t.term} (${t.exact})`)}:{kind:'filtered'};}
    return {status:matches.length?'matched':'empty',plan:exact?{strategy:'identifier'}:plan,matches,total:matches.length,nextCursor:null,...(corrections.length&&{corrections})};
  };
}
