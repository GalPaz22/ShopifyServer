import {createHash} from 'node:crypto';
// Content hash shared by the search engine; kept dependency-free so the engine can run without the crawling stack.
export function hash(value){
 const digest=createHash('sha256');
 // Catalogs are large JSON arrays. Serialize a bounded batch at a time rather than
 // allocating a second, catalog-sized Unicode string during a hot rebuild.
 // A custom toJSON may depend on its array index: retain native serialization there.
 if(!Array.isArray(value)||typeof value.toJSON==='function'||value.some(v=>typeof v?.toJSON==='function'))return digest.update(JSON.stringify(value)).digest('hex');
 digest.update('[');
 for(let start=0;start<value.length;start+=128){
  const parts=[];for(let i=start;i<Math.min(start+128,value.length);i++)parts.push(JSON.stringify(value[i])??'null');
  if(start)digest.update(',');digest.update(parts.join(','));
 }
 return digest.update(']').digest('hex');
}
