import test from 'node:test';import assert from 'node:assert/strict';import {createTenantRoutes,manifest} from './tenants/rikushet/index.mjs';
for(const strict of [true,false])test('empty autocomplete '+(strict?'stops legacy fallback':'preserves old-profile fallback'),async()=>{
 const search=async()=>{};search.suggest=async()=>({matches:[],revision:9,strictAlternatives:strict});let next=0,body,header;
 const route=createTenantRoutes({search,getDb:async()=>({collection:()=>({})})});
 await route.autocomplete({store:{dbName:manifest.dbName,semantix:{module:manifest.slug,enabled:true}},query:{query:'דרגות'}},{setHeader:(k,v)=>header=v,json:x=>body=x},()=>next++);
 assert.equal(next,strict?0:1);if(strict){assert.deepEqual(body,[]);assert.equal(header,'rikushet@9');}
});
