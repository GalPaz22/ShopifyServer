import{test}from'node:test';import assert from'node:assert/strict';
const{createSearchService}=await import(process.env.FOX_SEARCH_ENGINE||'./tenants/foxhome/engine/core/semantic.mjs');
const product=(id,title='אגרטל זכוכית')=>({id,title,tenantId:'foxhome',categories:['אגרטל'],collections:[],colors:[],finishes:[],tags:[],specifications:{},description:'כלי זכוכית למים',price:20,stockStatus:'instock',hidden:false});
const profile=(enabled=true)=>({tenantId:'foxhome',domain:'homeware',productTypes:{},queryAliases:{},semanticAliases:{},scopedAliases:[],colors:{},finishes:{},tagDefinitions:{},rankingRules:[],hookData:{catalogSearchRules:{alwaysResults:enabled,llmOnEmpty:enabled}}});
const empty=()=>({status:'empty',matches:[],total:0,plan:{terms:[]}}),planner={intent:'כלי להשרשת ענף במים',categories:['אגרטל'],terms:['זכוכית'],requirements:['כלי לייחור'],maxPrice:null,minPrice:null,colors:[],finishes:[],tags:[],clarification:''};
test('zero results after final tenant guards get LLM interpretation before catalog fallback',async()=>{
 const p=product('vase'),calls=[];let guards=0;
 const service=createSearchService([p],profile(),async req=>{calls.push(req.stage);return{model:'test',data:req.stage==='interpret'?planner:{matches:[],alternatives:[{id:p.id,reason:'כלי זכוכית להחזקת מים',missing:['לא אומתה התאמה לכל ייחור']}]}};},{expansion:'off',lightweightRouter:false,retrieve:()=>({status:'matched',matches:[p],total:1,plan:{terms:[]}}),hooks:{has:n=>n==='rerank',rerank:async list=>++guards===1?[]:list}});
 const r=await service({query:'ייחור'});assert.ok(calls.includes('interpret'));assert.ok(calls.includes('select'));assert.ok(r.matches.length);assert.equal(r.metadata.emptyResultRescue,true);assert.equal(r.metadata.llmUsed,true);assert.ok(r.matches.every(p=>p.matchQuality==='alternative'));
});
test('normal matched products stay fast and do not call a model',async()=>{
 const p=product('vase');let calls=0;const service=createSearchService([p],profile(),async()=>{calls++;throw Error('unneeded');},{expansion:'off',retrieve:()=>({status:'matched',matches:[p],total:1,plan:{terms:[]}})});assert.equal((await service({query:'אגרטל'})).total,1);assert.equal(calls,0);
});
test('unknown intent is read by the LLM and never ends at zero when the catalog is sellable',async()=>{
 let calls=0;const service=createSearchService([product('vase')],profile(),async req=>{calls++;return{model:'test',data:req.stage==='interpret'?{...planner,requirements:[],clarification:'אין צמח חי בחנות'}:{matches:[]}};},{retrieve:empty,lightweightRouter:false});const r=await service({query:'ייחורים'});assert.ok(calls>0);assert.ok(r.total>0);assert.equal(r.metadata.llmUsed,true);assert.equal(r.metadata.closestFallback,true);assert.ok(r.matches.every(p=>p.matchQuality==='alternative'));
});
test('model outages still return real catalog products with honest alternative labels',async()=>{
 let calls=0;const service=createSearchService([product('vase')],profile(),async()=>{calls++;throw Error('provider unavailable');},{retrieve:empty,lightweightRouter:false});const r=await service({query:'ייחורים'});assert.ok(calls>0);assert.equal(r.matches[0].id,'vase');assert.equal(r.matches[0].matchQuality,'alternative');assert.equal(r.metadata.fullMatch,false);
});
test('model cannot inject a product id absent from the supplied catalog',async()=>{
 const service=createSearchService([product('vase')],profile(),async req=>({model:'test',data:req.stage==='interpret'?planner:{matches:[],alternatives:[{id:'invented',reason:'made up',missing:[]}]}}),{retrieve:empty,lightweightRouter:false});const r=await service({query:'ייחור'});assert.ok(r.total>0);assert.ok(r.matches.every(p=>p.id==='vase'));
});
test('fallback after guard rejects all model picks stays labelled and bounded',async()=>{
 let calls=0;const p=product('vase');const service=createSearchService([p],profile(),async req=>{calls++;return{model:'test',data:req.stage==='interpret'?planner:{matches:[],alternatives:[{id:p.id,reason:'חלופה',missing:['לא מדויק']}]}};},{expansion:'off',lightweightRouter:false,retrieve:()=>({status:'matched',matches:[p],total:1,plan:{terms:[]}}),hooks:{has:n=>n==='rerank',rerank:async()=>[]}});const r=await service({query:'מראה'});assert.ok(r.total>0);assert.equal(r.matches[0].matchQuality,'alternative');assert.ok(calls<=3);
});
test('empty catalog never fabricates products and unrelated tenants keep their opt-out',async()=>{
 const s=createSearchService([],profile(),async()=>{throw Error('unavailable');},{retrieve:empty,lightweightRouter:false});assert.equal((await s({query:'ייחור'})).total,0);
 let calls=0;const p=product('vase'),off=createSearchService([p],profile(false),async()=>{calls++;throw Error('unneeded');},{expansion:'off',retrieve:()=>({status:'matched',matches:[p],total:1,plan:{terms:[]}}),hooks:{has:n=>n==='rerank',rerank:async()=>[]}});assert.equal((await off({query:'אגרטל'})).total,0);assert.equal(calls,0);
});
