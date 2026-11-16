import {normalize,planQuery,search,applySpelling,sellable,matchesTaxonomy} from './core.mjs';

const tokens=text=>normalize(text).match(/[\p{L}\p{N}]+/gu)||[];
export function distance(a,b){
  let row=Array.from({length:b.length+1},(_,i)=>i);
  for(let i=1;i<=a.length;i++){
    const next=[i];for(let j=1;j<=b.length;j++)next[j]=Math.min(next[j-1]+1,row[j]+1,row[j-1]+(a[i-1]===b[j-1]?0:1));row=next;
  }return row[b.length];
}

// Conservative spelling repair against title phrases, not descriptions.
// Never repairs numeric model/SKU tokens, short words, or long sentences.
export function createSpellingResolver(products,client){
  const visible=products.filter(p=>p.tenantId===client.tenantId&&sellable(p,client));
  const vocabulary=new Map();
  for(const p of visible){const words=tokens(p.title);
    for(let start=0;start<words.length;start++)for(let count=1;count<=3&&start+count<=words.length;count++){
      const phrase=words.slice(start,start+count).join(' '),compact=phrase.replaceAll(' ','');
      if(compact.length<6||compact.length>24||/\d/.test(compact))continue;
      if(!vocabulary.has(compact))vocabulary.set(compact,{compact,label:phrase,ids:new Set()});
      vocabulary.get(compact).ids.add(p.id);
    }
  }
  return query=>{
    const corrected=applySpelling(query,client.queryAliases);
    if(corrected!==normalize(query)){
      let page=search(products,client,{query:corrected,limit:50});
      const matches=[...page.matches];
      while(page.nextCursor){page=search(products,client,{cursor:page.nextCursor,limit:50});matches.push(...page.matches);}
      if(matches.length)return {status:'matched',matches,metadata:{mode:'spelling',llmUsed:false,llmCalls:0,correction:{from:query,to:corrected,source:'tenant-alias'}}};
    }
    const plan=planQuery(query,client),words=tokens(plan.terms.join(' '));
    if(!words.length||words.length>3)return null;
    const compact=words.join('');if(compact.length<6||compact.length>24||/\d/.test(compact))return null;
    const maxEdits=compact.length>=10?2:1;
    const candidates=[];
    for(const entry of vocabulary.values()){
      if(Math.abs(entry.compact.length-compact.length)>maxEdits)continue;
      const edits=distance(compact,entry.compact);if(edits<=maxEdits)candidates.push({...entry,edits});
    }
    candidates.sort((a,b)=>a.edits-b.edits||a.label.localeCompare(b.label));
    if(!candidates.length)return null;
    const best=candidates[0];
    // Equal-distance, dissimilar interpretations are ambiguous: defer to LLM.
    if(candidates.some(c=>c.edits===best.edits&&distance(c.compact,best.compact)>1))return null;
    const variants=candidates.filter(c=>distance(c.compact,best.compact)<=1);
    const ids=new Set(variants.flatMap(c=>[...c.ids]));
    const matches=visible.filter(p=>matchesTaxonomy(p,plan)&&ids.has(p.id)&&(!plan.productType||p.productType===plan.productType)&&
      plan.colors.every(c=>p.colors.includes(c))&&plan.finishes.every(f=>p.finishes.includes(f))&&plan.tags.every(t=>(p.tags||[]).includes(t))&&
      (plan.maxPrice===null||(p.price!==null&&p.price<=plan.maxPrice)));
    matches.sort((a,b)=>a.id.localeCompare(b.id));
    return matches.length?{status:'matched',matches,plan:{...plan,strategy:'catalog-spelling'},
      metadata:{mode:'spelling',llmUsed:false,llmCalls:0,correction:{from:plan.terms.join(' '),to:best.label,variants:variants.map(v=>v.label),edits:best.edits}}}:null;
  };
}
