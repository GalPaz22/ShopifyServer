import test from 'node:test';
import assert from 'node:assert/strict';
import {createSearchService} from './tenants/rikushet/engine/core/semantic.mjs';
const products=[{id:'1',tenantId:'t',title:'תיק גב לטיולים',description:'תיק גב לטיולים',categories:[],colors:[],finishes:[],tags:[],stockStatus:'instock',price:100,hidden:false,specifications:{}}];
const client={tenantId:'t',platform:'magento',version:'test',colors:{},finishes:{},productTypes:{},rankingRules:[],queryAliases:{},hookData:{rikushetSearchRules:{strictAlternatives:true}}};
const plan={intent:'מוצר',categories:[],terms:['תיק'],requirements:['דרגות'],maxPrice:null,minPrice:null,colors:[],finishes:[],tags:[],clarification:''};
test('Rikushet does not ask for or accept forced alternatives after selector rejects candidates',async()=>{
 const calls=[];const run=createSearchService(products,client,async({stage,prompt,schema})=>{calls.push(stage);if(stage==='select'){assert.ok(!schema.properties.alternatives);assert.ok(!prompt.includes('Even if none are similar'));}return {data:stage==='interpret'?plan:{matches:[],alternatives:[{id:'1',reason:'מילוי לא קשור',missing:['דרגות']}]}};},{lightweightRouter:false});
 const r=await run({query:'דרגות'});assert.deepEqual(calls,['interpret','select']);assert.equal(r.total,0);assert.deepEqual(r.matches,[]);assert.equal(r.metadata.closestFallback,false);
});
test('Rikushet does not turn a provider failure into arbitrary catalog products',async()=>{
 const run=createSearchService(products,client,async()=>{throw Error('offline')},{lightweightRouter:false});const r=await run({query:'שקיות חימום'});assert.equal(r.total,0);assert.equal(r.status,'degraded');assert.equal(r.metadata.strictAlternatives,true);
});
test('Rikushet concurrency fallback stays empty without a second model call',async()=>{
 const run=createSearchService(products,client,async()=>{throw Error('should not run')},{llmConcurrency:0,lightweightRouter:false});const r=await run({query:'דרגות'});assert.equal(r.total,0);assert.equal(r.metadata.failureReason,'llm-concurrency');
});
test('verified semantic matches and literal products remain available',async()=>{
 const run=createSearchService(products,client,async({stage})=>({data:stage==='interpret'?{...plan,requirements:['לטיולים']}:{matches:[{id:'1',evidence:[{requirement:0,field:'title',quote:'לטיולים'}]}]}}),{lightweightRouter:false});
 const r=await run({query:'לטיול ארוך'});assert.equal(r.matches[0].id,'1');assert.equal(r.matches[0].matchQuality,undefined);
 const literal=createSearchService(products,client,async()=>{throw Error('no model expected')},{expansion:'off'});assert.equal((await literal({query:'תיק גב'})).matches[0].id,'1');
});
test('the new policy is opt-in and does not change other tenants',async()=>{
 const run=createSearchService(products,{...client,hookData:{}},async({stage})=>({data:stage==='interpret'?plan:{matches:[],alternatives:[{id:'1',reason:'חלופה',missing:['דרגות']}]}}),{lightweightRouter:false});
 const r=await run({query:'דרגות'});assert.ok(r.matches.length);assert.equal(r.matches[0].matchQuality,'alternative');
});
