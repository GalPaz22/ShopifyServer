import {randomUUID} from 'node:crypto';
// Ranked result snapshots in Mongo (chunks of 50) so load-more works across server instances and restarts.
const TTL_MS=30*60*1000,expired=()=>Error('Search expired; start a new search'),ready=new WeakMap();
export function createSessions(collection,{now=Date.now}={}){
 async function index(){if(!ready.has(collection))ready.set(collection,collection.createIndex({expiresAt:1},{expireAfterSeconds:0}).catch(e=>{ready.delete(collection);throw e;}));await ready.get(collection);}
 return {
  async save(result,limit){
   if(result.total<=limit)return {...result,matches:result.matches.slice(0,limit),nextCursor:null};
   await index();const id=randomUUID(),expiresAt=new Date(now()+TTL_MS),{matches,...meta}=result;
   const docs=[];for(let i=0;i<matches.length;i+=50)docs.push({_id:id+':'+i/50,session:id,chunk:i/50,expiresAt,matches:matches.slice(i,i+50),...(i===0&&{meta:{...meta,total:matches.length}})});
   await collection.insertMany(docs);
   return {...result,matches:matches.slice(0,limit),total:matches.length,nextCursor:id+':'+limit};
  },
  async read(cursor,limit){
   const m=/^([0-9a-f-]{36}):(\d{1,6})$/.exec(String(cursor));if(!m)throw Error('Invalid cursor');const [,id,offsetText]=m,offset=Number(offsetText);
   const first=await collection.findOne({_id:id+':0'});if(!first||first.expiresAt<new Date(now()))throw expired();
   const matches=[];for(let c=Math.floor(offset/50);c<=Math.floor((offset+limit-1)/50);c++){const doc=c===0?first:await collection.findOne({_id:id+':'+c});if(doc)matches.push(...doc.matches.map((p,i)=>({p,i:c*50+i})));}
   const page=matches.filter(x=>x.i>=offset&&x.i<offset+limit).map(x=>x.p),total=first.meta.total;
   return {...first.meta,matches:page,total,nextCursor:offset+limit<total?id+':'+(offset+limit):null};
  }
 };
}
