import test from 'node:test';
import assert from 'node:assert/strict';
import {buildSearchIndex,createIndexRetriever} from './tenants/steimatzky/engine/core/search-index.mjs';
import {createSearchService} from './tenants/steimatzky/engine/core/semantic.mjs';

// A title search that finds only the title gets the same author's other books, then popular products of the title's
// narrowest department, after it (03–10.11.2026: 86% of title-match answers held 1–3 products, 27.1% vs 30.1% search→add).
const profile={tenantId:'t',productTypes:{},colors:{},finishes:{},tagDefinitions:{},queryAliases:{},badgeRules:[],pipeline:{outOfStock:'last',seriesOrder:true}};
const book=(id,title,extra={})=>({id,tenantId:'t',title,description:'',categories:[],tags:[],colors:[],finishes:[],stockStatus:'instock',hidden:false,price:60,specifications:{},...extra});
const setup=products=>createSearchService(products,profile,()=>{throw Error('no model call expected');},{retrieve:createIndexRetriever(products,profile,buildSearchIndex(products,'1')),lightweightRouter:false,expansion:'sparse',expandBelow:5});
const ids=r=>r.matches.map(p=>p.id);
const filler=(n,category)=>Array.from({length:n},(_,i)=>book('f'+i,'ספר מילוי מספר '+i,{categories:[category,'ספרים בעברית'],popularity:i}));

test('the title leads, then the author\'s other books, then the narrowest department by popularity',async()=>{
 const carle=(id,title,pop)=>book(id,title,{categories:['דפים עבים','ספרים בעברית'],specifications:{author:'אריק קרל'},popularity:pop});
 const products=[carle('z','הזחל הרעב-דפים עבים',140),carle('c2','מה הצבע שלך',10),carle('c3','הכוכבים של אבא',50),...filler(14,'דפים עבים'),book('broad','ספר כללי',{categories:['ספרים בעברית'],popularity:9999})];
 const r=await setup(products)({query:'הזחל הרעב',limit:12});
 assert.equal(r.metadata.phase,'title-match');
 assert.equal(ids(r)[0],'z');
 assert.deepEqual(ids(r).slice(1,3),['c3','c2']);
 assert.equal(r.matches[1].matchQuality,'related');
 assert.equal(ids(r)[3],'f13');
 assert.ok(!ids(r).includes('broad'));
 assert.equal(r.total,12);assert.equal(r.metadata.relatedByAuthor,2);
});

test('in-stock related products come before sold-out ones; hidden ones never come',async()=>{
 const a=(id,title,extra)=>book(id,title,{specifications:{author:'פרידה מקפדן'},categories:['מתח'],...extra});
 const r=await setup([a('own','אל תשקר',{}),a('s','הקולגה',{stockStatus:'outofstock',popularity:900}),a('i','ההתמחות',{popularity:5}),a('h','הסוד',{hidden:true,popularity:1000})])({query:'אל תשקר',limit:12});
 assert.deepEqual(ids(r),['own','i','s']);
});

test('a page of title matches is not widened; an author of hundreds of books adds none',async()=>{
 const many=Array.from({length:12},(_,i)=>book('m'+i,'הארי פוטר כרך '+i,{specifications:{author:'רולינג ג׳יי קיי'}}));
 const r=await setup([...many,book('o','ספר אחר',{specifications:{author:'רולינג ג׳יי קיי'}})])({query:'הארי פוטר',limit:12});
 assert.ok(r.matches.every(p=>p.matchQuality!=='related'));
 const prolific=Array.from({length:401},(_,i)=>book('p'+i,'כותר '+i,{specifications:{author:'מחבר פורה'}}));
 const r2=await setup([book('t','שיר הלילה',{specifications:{author:'מחבר פורה'}}),...prolific])({query:'שיר הלילה',limit:12});
 assert.deepEqual(ids(r2),['t']);
});

test('a sold-out product the search found stays before every related one',async()=>{
 const r=await setup([book('own','איתמר וכובע הקסמים השחור',{stockStatus:'outofstock',specifications:{author:'דויד גרוסמן'}}),book('o','איתמר פוגש ארנב',{specifications:{author:'דויד גרוסמן'},popularity:50})])({query:'כובע הקסמים',limit:12});
 assert.deepEqual(ids(r),['own','o']);
});

test('the department comes from any edition the words name, the narrowest one',async()=>{
 const r=await setup([book('ex','הזחל הרעב',{stockStatus:'outofstock',categories:['ספרים בעברית']}),book('board','הזחל הרעב-דפים עבים',{categories:['דפים עבים']}),...filler(13,'דפים עבים')])({query:'הזחל הרעב',limit:12});
 assert.deepEqual(ids(r).slice(0,2),['board','ex']);
 assert.ok(r.matches.slice(2).every(p=>p.matchQuality==='related'&&p.id.startsWith('f')));
 assert.equal(r.total,12);
});

test('the same title under another id is not repeated as a related product',async()=>{
 const r=await setup([book('a','סטונר',{specifications:{author:'ג׳ון ויליאמס'}}),book('b','אוגוסטוס',{specifications:{author:'ג׳ון ויליאמס'}}),book('b2','אוגוסטוס',{specifications:{author:'ג׳ון ויליאמס'}})])({query:'סטונר',limit:12});
 assert.deepEqual(ids(r),['a','b']);
});
