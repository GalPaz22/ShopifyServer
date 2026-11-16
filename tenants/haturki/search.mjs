import {readFileSync,existsSync} from 'node:fs';
import {gunzip} from 'node:zlib';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
import {createDraftRuntime} from './engine/runtime.mjs';
import {processProduct,retiredRows,feedCategories} from './engine/core/core.mjs';
import {hash} from './engine/core/hash.mjs';
import {buildSearchIndexAsync} from './engine/core/search-index.mjs';
import {createSessions} from './sessions.mjs';
const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));
const unzip=promisify(gunzip);
export const manifest=read('./semantix.module.json');
const LIVE=["price","regularPrice","stockStatus","hidden","status","url","image"];
const strings=v=>Array.isArray(v)?v.map(x=>typeof x==='string'?x:x?.name||x?.title||x?.label).filter(x=>typeof x==='string'):typeof v==='string'&&v?[v]:[];
const number=v=>v===null||v===undefined||v===''?null:Number.isFinite(Number(v))?Number(v):null;
const authorOf=row=>[row.author,row.siteCard?.author,row.specifications?.author].find(v=>typeof v==='string'&&v.trim())?.trim();
// Merchant-row fields the engine card lacks but a storefront needs: the platform's own item id (add to cart, tracking),
// its catalog number, the subtitle, the sale flag and sale price, the feed's offer id and the per-format price labels
// a card template prints (specialLabel: printed / digital / club). Only these ride along; the engine's own fields win on a clash.
const SOURCE_FIELDS=['ItemID','Makat','Subtitle','onSale','offerId','salePrice','specialLabel'];
const sources=new WeakMap();
export function withSource(rows){
 let byId=sources.get(rows);
 if(!byId){byId=new Map();for(const r of rows){const s={};for(const k of SOURCE_FIELDS)if(r[k]!==undefined&&r[k]!==null)s[k]=r[k];if(Object.keys(s).length)byId.set(String(r.id??r._id),s);}sources.set(rows,byId);}
 return p=>{const s=byId.get(String(p.id))||byId.get(String(p.id).split(':').pop());return s?{...s,...p}:p;};
}
// A merchant Mongo row in the engine's raw shape (as the studio imports existing clients).
export function adaptRow(row,retired=()=>false){
 const specifications=row.specifications&&typeof row.specifications==='object'&&!Array.isArray(row.specifications)?row.specifications:{};
 const author=authorOf(row);
 return {id:String(row.id??row._id),name:row.name||row.title||'',sku:String(row.sku??'').trim(),url:row.url||row.permalink,image:row.image||row.images?.[0]?.src||row.images?.[0]||null,
  price:number(row.price),regularPrice:number(row.regularPrice??row.regular_price),currency:row.currency||null,
  stockStatus:row.stockStatus||row.stock_status||'unknown',status:row.status||'ACTIVE',
  // Hidden as the studio imports it: the store's own flag, "not in store", a catalog visibility of "hidden", or an
  // row the store has retired (dead product page, never seen by a sync, or out of stock with no price / missed by the latest sync — retiredRows).
  hidden:row.hidden===true||row.notInStore===true||row.catalog_visibility==='hidden'||retired(row),
  ...(row.collections!==undefined&&{collections:strings(row.collections)}),categories:(c=>c.length?c:feedCategories(row))(strings(row.categories?.length?row.categories:row.category)),tags:[...new Set([...strings(row.tags),...strings(row.siteTags),...(row.isWooPopular===true?['isWooPopular']:[])])],
  // WooCommerce's popular flag and sales count, read live: the engine lifts popular products by default, best sellers first.
  ...(typeof row.isWooPopular==='boolean'&&{popular:row.isWooPopular}),...(Number(row.total_sales)>0&&{popularity:Number(row.total_sales)}),
  sourceContentHash:row.sourceContentHash||null,description:String(row.description||row.short_description||'').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim().slice(0,18000),specifications:author?{...specifications,author}:specifications};
}
// Enriched snapshot cards with live commercial fields; new products are processed with the same profile. seen collects
// the ids of the live rows the cards cover.
export function mergeLive(cards,rows,profile,tenantId,seen=new Set()){
 const retired=retiredRows(rows),live=new Map(rows.map(r=>{const a=adaptRow(r,retired);if(r.categories===undefined&&r.category===undefined&&!feedCategories(r).length)delete a.categories;return [a.id,a];})),out=[];
 for(const c of cards){const key=String(c.id).split(':').pop(),row=live.get(String(c.id))||live.get(key);if(!row)continue;seen.add(row.id);
  const next={...c};for(const k of LIVE)if(row[k]!==undefined&&row[k]!==null)next[k]=row[k];
  // A full sync owns source text; keep derived specifications/tags and the
  // tenant's rules, but do not keep searching the old product name/description.
  if(row.sourceContentHash&&row.sourceContentHash!==c.sourceContentHash){
   Object.assign(next,{title:row.name,description:row.description,sourceContentHash:row.sourceContentHash,specifications:{...c.specifications,...row.specifications},tags:[...new Set([...(c.tags||[]),...row.tags])]});
  }
  if(row.sku)next.sku=row.sku;
  if(row.specifications.author)next.specifications={...next.specifications,author:row.specifications.author};
  // The store's popular flag and sales follow the live rows, so a newly popular product is lifted without a new edition.
  if(row.popular!==undefined){const tags=(next.tags||[]).filter(t=>t!=='isWooPopular');next.tags=row.popular?[...tags,'isWooPopular']:tags;}
  if(row.popularity)next.popularity=row.popularity;for(const field of ['categories','collections'])if(row[field]!==undefined)next[field]=row[field];out.push(next);}
 const client={...profile,tenantId,version:'semantix-mini',publishedStatuses:['ACTIVE','publish']};
 for(const [id,row] of live)if(!seen.has(id))out.push({...processProduct(row,client),sku:row.sku,sourceContentHash:row.sourceContentHash,...(row.popularity&&{popularity:row.popularity})});
 return out;
}
// The module's data (approved revision: profile + enriched cards) is published by the studio to the store's own
// database: semantix_module → "<slug>:head" points at a gzip edition stored in parts. The head is one _id lookup,
// checked at most every checkMs; an edition is downloaded only when its digest changes. Local profile.json/snapshot.json
// (a downloaded ZIP) are used only while nothing is published. A failed check keeps serving the edition in memory.
// An edition keeps its parsed cards only until the build that uses them: the runtime then holds its own copy, and a
// later live rebuild downloads the edition again by digest (or rereads the local snapshot.json).
export async function loadEdition(store,head){
 const parts=await store.find({kind:'part',slug:manifest.slug,digest:head.digest}).toArray();
 if(parts.length!==head.parts)throw Error('Published module is incomplete');
 const bytes=v=>v?._bsontype==='Binary'?Buffer.from(v.buffer):Buffer.from(v);
 const gz=Buffer.concat(parts.sort((a,b)=>a.part-b.part).map(p=>bytes(p.data)));
 if(createHash('sha256').update(gz).digest('hex').slice(0,24)!==head.digest)throw Error('Published module is corrupt');
 return {...JSON.parse((await unzip(gz)).toString('utf8')),digest:head.digest,parts:head.parts};
}
const snapshotOf=async(e,store)=>e.snapshot??(e.digest==='local'?read('./snapshot.json'):(await loadEdition(store,e)).snapshot);
const local=()=>{const has=n=>existsSync(new URL(n,import.meta.url));return has('./profile.json')&&has('./snapshot.json')?{revision:manifest.revision,profile:read('./profile.json'),digest:'local'}:null;};
// The live-rows part of the fingerprint, once per rows array (the loader returns the same array until it refreshes,
// and a refresh is a new array): hashing the whole catalog on every search and suggestion cost CPU and garbage.
const prints=new WeakMap();
const rowsPrint=rows=>{let p=prints.get(rows);if(p===undefined)prints.set(rows,p=hash([rows.catalogVersion,rows.map(r=>[r.id??r._id,r.sourceContentHash,r.sku??null,authorOf(r)??null,...LIVE.map(k=>r[k]??r[k==='stockStatus'?'stock_status':k]??null),r.catalog_visibility??null,r.notInStore??null,r.lastSeenAt??null,r.pageCheck?.status??null,r.isWooPopular??null,r.total_sales??null,r.categories??null,r.category??null,r.productTypes??null,r.customAttributes??null,r.collections??null])]));return p;};
// Live rows the built edition already covers with enriched cards are cut down to what the fingerprint, the live fields,
// the hidden rule and withSource read; their text (description, specifications, images...) stays only in the runtime.
// Rows the edition lacks keep everything: a build processes them as new products. Only the module's own row loader
// (it can reload) has its rows slimmed, in place, so a caller's own array is never changed.
const SLIM_FIELDS=['_id','id','sourceContentHash','sku','author','price','regularPrice','regular_price','currency','stockStatus','stock_status','status','hidden','notInStore','catalog_visibility','lastSeenAt','pageCheck','isWooPopular','total_sales','url','permalink','image','categories','category','collections','productTypes','customAttributes',...SOURCE_FIELDS];
const slim=new WeakSet();
function slimRow(r){const s={};for(const k of SLIM_FIELDS)if(r[k]!==undefined)s[k]=r[k];const author=authorOf(r);if(author)s.author=author;if(!r.categories?.length&&!r.category?.length){const feed=feedCategories(r);if(feed.length)s.productTypes=feed;}if(!r.image&&r.images?.length)s.images=[r.images[0]];slim.add(s);return s;}
function compact(rows,covered){for(let i=0;i<rows.length;i++){const r=rows[i];if(!slim.has(r)&&covered.has(String(r.id??r._id)))rows[i]=slimRow(r);}}
// A slimmed row this edition has no card for would be processed from its live fields alone.
function uncovered(cards,rows){if(!rows.some(r=>slim.has(r)))return false;const ids=new Set();for(const c of cards){ids.add(String(c.id));ids.add(String(c.id).split(':').pop());}
 return rows.some(r=>slim.has(r)&&(r.sourceContentHash||!ids.has(String(r.id??r._id))));}
// liveRebuildMs: a rebuild of the whole runtime costs seconds of CPU on a large catalog and blocks every search on the
// server meanwhile, so live price/stock changes are folded in twice a day (a new edition is built at once).
export function createTenantSearch({loadRows,checkMs=30000,liveRebuildMs=43200000,now=Date.now,createRuntime=createDraftRuntime,onLifecycle=()=>{}}={}){
 let fingerprint=null,runtime=null,edition=null,checkedAt=-Infinity,checking=null,building=null,builtDigest=null,builtAt=-Infinity,covered=null,builtCatalogVersion=null;
 async function published(store){
  if(store&&now()-checkedAt>=checkMs)checking??=(async()=>{
   try{const head=await store.findOne({_id:manifest.slug+':head'},{projection:{digest:1,parts:1,revision:1}});
    if(head&&head.digest!==edition?.digest)edition=await loadEdition(store,head);else if(!head&&edition?.digest!=='local')edition=null;checkedAt=now();}
   catch(e){if(!edition)throw e;console.error('[SEMANTIX '+manifest.slug+'] keeping revision',edition.revision,'-',e.message);checkedAt=now();}
   finally{checking=null;}})();
  // A slow module download must not hold requests for an already working edition.
  if(checking&&!runtime)await checking;
  return edition??=local()??(()=>{throw Error('No published module data');})();
 }
 async function current(collection,store){
  const e=await published(store),rows=await loadRows(collection,store),print=e.digest+':'+rowsPrint(rows);
  // The first build answers this request. Later changes (a new edition, live price or stock) are built in the background
  // while the previous runtime keeps answering, so no shopper waits for a rebuild of the whole catalog.
  if(!runtime)await (building??=build(e,rows,collection,store).finally(()=>{building=null;}));
  else if(print!==fingerprint&&!building&&(e.digest!==builtDigest||rows.catalogVersion&&rows.catalogVersion!==builtCatalogVersion||now()-builtAt>=liveRebuildMs))building=new Promise(r=>setTimeout(r,0)).then(()=>build(e,rows,collection,store))
   .catch(err=>console.error('[SEMANTIX '+manifest.slug+'] rebuild failed, keeping revision',runtime.revision,'-',err.message)).finally(()=>{building=null;});
  // Rows refreshed since the last build are slimmed against the edition that build used.
  if(loadRows.reload&&covered&&builtDigest===e.digest&&!building)compact(rows,covered);
  runtime.attach=withSource(rows);
  return runtime;
 }
 async function build(e,rows,collection,store){
  const s=await snapshotOf(e,store);
  // Rows slimmed against an earlier edition that this one does not carry are read again in full.
  if(loadRows.reload&&uncovered(s.productCards,rows))rows=await loadRows.reload(collection);
  const seen=new Set(),cards=mergeLive(s.productCards,rows,e.profile,manifest.tenantId,seen);
  const priorIndex=runtime?.index,contentHash=hash(cards);
  const index=priorIndex?.contentHash===contentHash?priorIndex:await buildSearchIndexAsync(cards,'studio-'+e.revision);
  const next=createRuntime({id:manifest.tenantId,url:manifest.url,platform:manifest.platform,productCards:cards,productCardsProfileHash:hash(e.profile),searchIndex:index,storeContext:s.storeContext,tagAssignments:s.tagAssignments,studioVectors:s.studioVectors,catalog:{products:[]}},{number:e.revision,profile:e.profile},{releaseBase:true});
  await next.search.ready?.();
  onLifecycle({stage:'ready',revision:e.revision,indexReused:index===priorIndex,products:cards.length});
  next.revision=e.revision;next.attach=withSource(rows);runtime=next;fingerprint=e.digest+':'+rowsPrint(rows);builtDigest=e.digest;builtCatalogVersion=rows.catalogVersion;builtAt=now();covered=seen;delete e.snapshot;
  if(loadRows.reload)compact(rows,seen);
 }
 // Page size when the storefront sends no limit: the tenant's pipeline.pageSize (published with the profile), else 12.
 const search=async function({collection,sessions,moduleStore,request}){
  const limit=request.limit??(await published(moduleStore)).profile?.pipeline?.pageSize??12;if(!Number.isInteger(limit)||limit<1||limit>50)throw Error('Invalid limit');
  if(request.cursor){if(request.query!==undefined)throw Error('Invalid request');if(!sessions)throw Error('Search expired; start a new search');return createSessions(sessions).read(request.cursor,limit);}
  const rt=await current(collection,moduleStore);let r=await rt.search({query:request.query,limit:50});const matches=[...r.matches];
  while(r.nextCursor&&matches.length<500){r=await rt.search({cursor:r.nextCursor,limit:50});matches.push(...r.matches);}
  const attached=matches.map(rt.attach),result={...r,matches:attached,total:matches.length,nextCursor:null,metadata:{...r.metadata,searchEngine:'semantix-'+manifest.slug,revision:rt.revision}};
  return sessions?createSessions(sessions).save(result,limit):{...result,matches:attached.slice(0,limit)};
 };
 // Suggestions while the shopper types: the same edition and live rows as search, answered from the index only.
 search.suggest=async({collection,moduleStore,query,limit})=>{
  const rt=await current(collection,moduleStore);if(typeof rt.suggest!=='function')throw Error('Engine has no suggestions; export the module again');
  const r=await rt.suggest(query,{limit});return {...r,matches:r.matches.map(rt.attach),revision:rt.revision};
 };
 return search;
}
// Default Mongo loader: the merchant's products collection. Only the first load makes a search wait: rows older than ttlMs
// are refreshed in the background while the cached ones keep answering (a failed refresh keeps them). Every refresh
// decodes the whole collection; it runs twice a day, like the live rebuild that uses it.
export function createRowLoader({ttlMs=43200000,checkMs=15000,now=Date.now,max=100000}={}){
 let cached=null,expires=0,pending=null,checkedAt=-Infinity,checking=null,version=null;
 const check=store=>!store||now()-checkedAt<checkMs?null:checking??=(async()=>{try{const head=await store.findOne({_id:'catalog:head'},{projection:{catalogVersion:1}});const next=head?.catalogVersion||null;if(next!==version){version=next;expires=0;}checkedAt=now();}catch(e){if(!cached)throw e;checkedAt=now();}finally{checking=null;}})();
 const refresh=collection=>{if(now()>=expires&&!pending)load(collection).catch(e=>console.error('[SEMANTIX '+manifest.slug+'] catalog refresh failed -',e.message));};
 // Only the first load waits for the catalog version; afterwards it is checked beside the searches, holding none.
 const get=async(collection,store)=>{
  if(!cached){await check(store);return load(collection);}
  refresh(collection);check(store)?.then(()=>refresh(collection));
  return cached;
 };
 // A fresh full read, for a build that finds rows slimmed against another edition.
 get.reload=collection=>load(collection);
 return get;
 function load(collection){
  if(pending)return pending;
  const seen=version;
  pending=(async()=>{const cursor=collection.find({},{projection:{_id:1,id:1,name:1,title:1,sku:1,author:1,'siteCard.author':1,url:1,permalink:1,image:1,images:1,price:1,regularPrice:1,regular_price:1,currency:1,stockStatus:1,stock_status:1,status:1,hidden:1,notInStore:1,catalog_visibility:1,sourceContentHash:1,lastSeenAt:1,pageCheck:1,productTypes:1,customAttributes:1,isWooPopular:1,total_sales:1,categories:1,category:1,collections:1,tags:1,siteTags:1,description:1,short_description:1,specifications:1,...Object.fromEntries(SOURCE_FIELDS.map(k=>[k,1]))},maxTimeMS:20000}).limit(max+1).batchSize(500);
   try{const rows=await cursor.toArray();if(rows.length>max)throw Error('Catalog exceeds safety limit');Object.defineProperty(rows,'catalogVersion',{value:seen});cached=rows;expires=version===seen?now()+ttlMs:0;return rows;}finally{await cursor.close?.();}})()
   .finally(()=>{pending=null;});
  return pending;
 }
}
