import {GoogleGenAI} from '@google/genai';
export const embeddingModel=()=>process.env.STUDIO_EMBEDDING_MODEL||'gemini-embedding-2';
export async function embedText(text,{query=false,model=embeddingModel()}={}) {
  const apiKey=process.env.GEMINI_API_KEY||process.env.GOOGLE_API_KEY;if(!apiKey)throw Error('נדרש מפתח מודל ליצירת וקטורים');
  const old=model.includes('001');
  const contents=old?text:query?'task: search result | query: '+text:'title: Product | text: '+text;
  const result=await new GoogleGenAI({apiKey}).models.embedContent({model,contents,config:{outputDimensionality:768,...(old?{taskType:query?'RETRIEVAL_QUERY':'RETRIEVAL_DOCUMENT'}:{}),httpOptions:{timeout:45000,retryOptions:{attempts:1}}}});
  const vector=result.embeddings?.[0]?.values;if(!Array.isArray(vector)||vector.length!==768||!vector.every(Number.isFinite)||!vector.some(v=>v!==0))throw Error('וקטור לא תקין');return {vector,model};
}
export const documentText=p=>[p.title,p.description,...p.categories,...(p.collections||[]),...p.tags,...Object.entries(p.specifications||{}).map(([k,v])=>k+': '+v)].join('\n').slice(0,20000);
export function cosine(a,b){if(a.length!==b.length)throw Error('ממדי וקטור אינם תואמים');let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return aa&&bb?dot/Math.sqrt(aa*bb):0;}
export function createVectorRanker(index,embed=embedText){return async(query,eligible)=>{const result=await embed(query,{query:true,model:index.model});return eligible.map(p=>({p,score:index.vectors[p.id]?cosine(result.vector,index.vectors[p.id]):0})).sort((a,b)=>b.score-a.score||a.p.id.localeCompare(b.p.id));};}

// Studio vectors: many products per request, compact storage (Float32, base64) keyed by product id with a text hash,
// kept apart from the build's vectorIndex so later processing does not throw them away.
export async function embedBatch(texts,{query=false,model=embeddingModel(),dimensions=256}={}){
  const apiKey=process.env.GEMINI_API_KEY||process.env.GOOGLE_API_KEY;if(!apiKey)throw Error('נדרש מפתח מודל ליצירת וקטורים');
  // One Content per text: a plain string array is read by gemini-embedding-2 as ONE multi-part input (one vector).
  const contents=texts.map(t=>({parts:[{text:query?'task: search result | query: '+t:'title: Product | text: '+t}]}));
  const result=await new GoogleGenAI({apiKey}).models.embedContent({model,contents,config:{outputDimensionality:dimensions,httpOptions:{timeout:90000,retryOptions:{attempts:2}}}});
  const vectors=(result.embeddings||[]).map(e=>e.values);
  if(vectors.length!==texts.length||vectors.some(v=>!Array.isArray(v)||v.length!==dimensions))throw Error('תשובת וקטורים לא תקינה');return vectors;
}
export function packVectors(entries,{model,dimensions}){
  const ids=[],hashes=[],data=new Float32Array(entries.length*dimensions);
  entries.forEach(([id,textHash,vector],i)=>{ids.push(id);hashes.push(textHash);data.set(vector,i*dimensions);});
  return {model,dimensions,ids,hashes,data:Buffer.from(data.buffer).toString('base64'),builtAt:new Date().toISOString()};
}
const unpacked=new WeakMap();
export function unpackVectors(store){
  if(!unpacked.has(store)){const buf=Buffer.from(store.data,'base64'),all=new Float32Array(buf.buffer,buf.byteOffset,buf.byteLength/4),map=new Map();
    store.ids.forEach((id,i)=>map.set(id,all.subarray(i*store.dimensions,(i+1)*store.dimensions)));unpacked.set(store,map);}
  return unpacked.get(store);
}
export function mergeVectors(store,entries,{model,dimensions}){
  const map=new Map();
  if(store&&store.model===model&&store.dimensions===dimensions){const vectors=unpackVectors(store);store.ids.forEach((id,i)=>map.set(id,[store.hashes[i],vectors.get(id)]));}
  for(const [id,h,v] of entries)map.set(id,[h,v]);
  return packVectors([...map].map(([id,[h,v]])=>[id,h,v]),{model,dimensions});
}
export function createPackedRanker(store,embed=async q=>(await embedBatch([q],{query:true,model:store.model,dimensions:store.dimensions}))[0]){
  return async(query,eligible)=>{const q=await embed(query),map=unpackVectors(store);return eligible.map(p=>{const v=map.get(p.id);return {p,score:v?cosine(q,v):0};}).sort((a,b)=>b.score-a.score||a.p.id.localeCompare(b.p.id));};
}
