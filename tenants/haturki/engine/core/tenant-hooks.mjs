import {Worker} from 'node:worker_threads';
import {createHash} from 'node:crypto';
import {normalize} from './core.mjs';

// Per-tenant functions the studio agent writes for ONE merchant, called by the search engine:
//   rewriteQuery(query, ctx)        → string            before retrieval
//   transformProduct(product, ctx)  → product           once per product when the catalog loads (tags, fields, cleanup…)
//   rerank(matches, query, ctx)     → id[]              final order; ids left out are dropped
// ctx = {tenant, data, normalize(s), words(s)}; data is the tenant's own JSON (lookup tables, word lists, weights…).
// Isolation, in layers: the source is checked for escape routes (no this/import/require/process/eval/async…), it runs in
// a worker thread with its own memory limit, inside an empty VM context that cannot compile strings, with a time limit per
// call; data crosses only as JSON text. A call that fails, times out or exhausts memory is reported and the search goes on
// without that function. The functions are part of the tenant's profile: versioned, verified and reviewable.
export const HOOKS={
 rewriteQuery:'function rewriteQuery(query, ctx) → string (the query to search). Runs before retrieval on every search.',
 transformProduct:'function transformProduct(product, ctx) → product. Runs once per product when the catalog loads; adds searchable knowledge: tags, categories, specifications, description, popularity, brand. What shoppers see and buy — title, image, id, url, price, stockStatus, hidden — always stays as in the source; put extra search words in tags or specifications.',
 rerank:'function rerank(matches, query, ctx) → array of ids in the order to show; ids left out are removed. matches: [{id,title,categories,collections,tags,specifications,brand,price,popularity,description(300 chars)}] in current order.',
};
const FORBIDDEN=/\b(import|require|process|globalThis|global|module|exports|eval|Function|constructor|__proto__|prototype|Reflect|Proxy|WebAssembly|setTimeout|setInterval|setImmediate|queueMicrotask|Atomics|SharedArrayBuffer|fetch|XMLHttpRequest|this|async|await|with|arguments|caller|callee|Symbol|getPrototypeOf|defineProperty|__defineGetter__|__lookupGetter__)\b/;
export const MAX_HOOK_SOURCE=20000,MAX_HOOK_DATA=200000;
const PROTECTED=['id','url','price','regularPrice','stockStatus','hidden','tenantId','sku','variants','title','image','images'];

export function validateHook(name,code){
 if(!Object.hasOwn(HOOKS,name))throw Error(`פונקציה לא מוכרת: ${name}. זמינות: ${Object.keys(HOOKS).join(', ')}`);
 if(typeof code!=='string'||!code.trim()||code.length>MAX_HOOK_SOURCE)throw Error(`קוד חסר או ארוך מ־${MAX_HOOK_SOURCE} תווים`);
 const bare=code.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g,' ').replace(/(["'`])(?:\\.|(?!\1)[^\\])*\1/g,'""');
 const bad=bare.match(FORBIDDEN);if(bad)throw Error(`הקוד משתמש ב־״${bad[0]}״, שאסור בפונקציות לקוח (פונקציות סינכרוניות וטהורות בלבד: בלי this, import, async, eval)`);
 if(!new RegExp(`function\\s+${name}\\s*\\(`).test(bare))throw Error(`הקוד חייב להגדיר function ${name}(…)`);
 return code;
}
export function validateHooks(hooks,data){
 if(hooks===undefined)return;
 if(!hooks||typeof hooks!=='object'||Array.isArray(hooks))throw Error('פונקציות לקוח לא תקינות');
 for(const [name,h] of Object.entries(hooks)){if(!h||typeof h!=='object')throw Error('פונקציה לא תקינה: '+name);validateHook(name,h.code);}
 if(data!==undefined&&(!data||typeof data!=='object'||Array.isArray(data)||JSON.stringify(data).length>MAX_HOOK_DATA))throw Error('hookData חייב להיות אובייקט JSON עד 200KB');
}
export const hasHooks=profile=>!!profile?.hooks&&Object.values(profile.hooks).some(h=>h?.code);

function source(hooks,data,tenant){
 const names=Object.keys(hooks);
 return `"use strict";
const normalize=${normalize.toString()};
const words=s=>normalize(s).split(' ').filter(Boolean);
const ctx=Object.freeze({tenant:${JSON.stringify(String(tenant))},data:JSON.parse(${JSON.stringify(JSON.stringify(data||{}))}),normalize,words});
const fns={};
${names.map(n=>`fns.${n}=(()=>{${hooks[n].code}\nreturn ${n};})();`).join('\n')}
var __run=(name,input)=>{const args=JSON.parse(input);const out=name==='transformProduct'?args[0].map(p=>fns.transformProduct(p,ctx)):fns[name](...args,ctx);return JSON.stringify(out===undefined?null:out);};`;
}

// Runners are shared by source (the studio creates many runtimes for one profile); old ones are closed.
const runners=new Map(),MAX_RUNNERS=12;
export function createHookRunner(hooks,data={},{tenant='',timeoutMs=100,batchTimeoutMs=8000,memoryMb=64}={}){
 const active=Object.fromEntries(Object.entries(hooks||{}).filter(([n,h])=>Object.hasOwn(HOOKS,n)&&h?.code));const names=Object.keys(active);if(!names.length)return null;
 for(const n of names)validateHook(n,active[n].code);
 const src=source(active,data,tenant),key=createHash('sha256').update(src).digest('hex');
 if(runners.has(key)){const r=runners.get(key);runners.delete(key);runners.set(key,r);return r;}
 let worker=null,seq=0;const pending=new Map();
 const fail=error=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(error);}pending.clear();};
 function spawn(){
  const w=new Worker(new URL('./hook-worker.mjs',import.meta.url),{workerData:{source:src},env:{},execArgv:[],resourceLimits:{maxOldGenerationSizeMb:memoryMb,maxYoungGenerationSizeMb:16,stackSizeMb:4}});
  worker=w;
  w.on('message',({id,out,error})=>{const p=pending.get(id);if(!p)return;pending.delete(id);clearTimeout(p.timer);error?p.reject(Error(error)):p.resolve(out);});
  w.on('error',e=>{if(worker===w){worker=null;fail(Error('הפונקציה קרסה: '+e.message));}});
  w.on('exit',()=>{if(worker===w){worker=null;fail(Error('הפונקציה נעצרה'));}});
  // After the listeners (adding them re-references the thread): an idle function worker never keeps the server alive.
  w.unref();
 }
 function call(name,args,timeout){
  if(!worker)spawn();const id=++seq,w=worker;
  const input=JSON.stringify(args);
  // Broad collection reranks include megabytes of source attributes. JSON parsing runs inside the
  // same VM deadline as the merchant function; allow 100ms per full MiB, capped at 900ms extra.
  const budget=timeout+(name==='rerank'?Math.min(900,Math.floor(input.length/1048576)*100):0);
  return new Promise((resolve,reject)=>{
   // The VM limit stops a runaway loop inside the worker; this outer limit also covers a stuck or memory-starved worker.
   const timer=setTimeout(()=>{pending.delete(id);reject(Error(`הפונקציה ${name} חרגה מזמן הריצה`));w.terminate();if(worker===w)worker=null;},budget+1000);
   pending.set(id,{resolve:out=>{try{resolve(JSON.parse(out));}catch{reject(Error('הפונקציה החזירה ערך לא תקין'));}},reject,timer});
   w.postMessage({id,name,input,timeout:budget});
  });
 }
 const runner={
  names,key,
  has:name=>names.includes(name),
  async rewriteQuery(query){const q=await call('rewriteQuery',[query],timeoutMs);if(typeof q!=='string'||q.length>300)throw Error('rewriteQuery חייב להחזיר מחרוזת עד 300 תווים');return q.trim()||query;},
  // Transform the catalog in order; protected fields always come from the source product.
  async transformProducts(products){
   // A full enriched catalog can be larger than the worker heap (Shipi: >80 MB). Bound each JSON message while
   // retaining the catalog-wide deadline and returning nothing until every batch has passed validation.
   const out=[],deadline=Date.now()+batchTimeoutMs;
   let batch=[],bytes=0;
   const flush=async()=>{
    if(!batch.length)return;
    const remaining=deadline-Date.now();if(remaining<=0)throw Error('הפונקציה transformProduct חרגה מזמן הריצה');
    const result=await call('transformProduct',[batch],remaining);
    if(!Array.isArray(result)||result.length!==batch.length)throw Error('transformProduct חייב להחזיר מוצר לכל מוצר');
    out.push(...result);batch=[];bytes=0;
   };
   for(const p of products){const size=JSON.stringify(p).length;
    if(batch.length&&(batch.length>=50||bytes+size>256000))await flush();
    batch.push(p);bytes+=size;
   }
   await flush();
   return out.map((p,i)=>{const src=products[i];if(!p||typeof p!=='object'||Array.isArray(p))throw Error('transformProduct החזיר ערך שאינו מוצר: '+src.id);
    const merged={...src,...p};for(const k of PROTECTED)if(Object.hasOwn(src,k))merged[k]=src[k];
    for(const k of ['categories','collections','tags'])merged[k]=Array.isArray(merged[k])?merged[k].filter(x=>typeof x==='string').slice(0,100):src[k]||[];
    if(!merged.specifications||typeof merged.specifications!=='object'||Array.isArray(merged.specifications))merged.specifications=src.specifications||{};
    merged.title=String(merged.title??src.title??'').slice(0,500);return merged;});
  },
  async rerank(matches,query){
   const brief=matches.map(p=>({id:p.id,title:p.title,categories:p.categories,collections:p.collections,tags:p.tags,specifications:p.specifications,brand:p.brand,price:p.price,popularity:p.popularity||0,description:String(p.description||'').slice(0,300)}));
   const ids=await call('rerank',[brief,query],timeoutMs);if(!Array.isArray(ids))throw Error('rerank חייב להחזיר מערך מזהים');
   const byId=new Map(matches.map(p=>[p.id,p])),seen=new Set(),out=[];for(const id of ids){const p=byId.get(String(id));if(p&&!seen.has(p.id)){seen.add(p.id);out.push(p);}}
   return out;
  },
  close(){runners.delete(key);fail(Error('closed'));worker?.terminate();worker=null;},
 };
 runners.set(key,runner);while(runners.size>MAX_RUNNERS)runners.values().next().value.close();
 return runner;
}
