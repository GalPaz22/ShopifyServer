import {normalize} from './core.mjs';

const hebrew=/^[א-ת]+$/u;
// Only internal vowel letters may vary. Initial/final י and ו can be consonants,
// and short words and identifiers must retain their literal meaning.
const key=word=>hebrew.test(word)&&word.length>=3?word[0]+word.slice(1,-1).replace(/[יו]/g,'')+word.at(-1):null;
const forms=word=>[word,...(hebrew.test(word)&&word.length>=4?[...'והבלמשכ'].map(p=>p+word):[])];
const edits=(a,b)=>{
  let row=Array.from({length:b.length+1},(_,j)=>j===0?0:/^[יו]+$/.test(b.slice(0,j))?j:Infinity);
  for(let i=1;i<=a.length;i++){
    const next=[/^[יו]+$/.test(a.slice(0,i))?i:Infinity];
    for(let j=1;j<=b.length;j++)next[j]=Math.min(a[i-1]===b[j-1]?row[j-1]:Infinity,/[יו]/.test(a[i-1])?row[j]+1:Infinity,/[יו]/.test(b[j-1])?next[j-1]+1:Infinity);
    row=next;
  }
  return row[b.length];
};

// A spelling is supported by the product's OWN title, before any appended author.
// AND all query words together: no single-word broadening or description evidence.
// The title vocabulary is built lazily and reused by search and autocomplete.
export function createTitleOrthographyResolver(products){
  let literal,variants;
  function prepare(){
    literal=new Map();variants=new Map();
    for(const p of products)for(const word of new Set(normalize(String(p.title).split(' / ')[0]).split(' ').filter(Boolean))){
      if(!literal.has(word)){
        literal.set(word,[]);const k=key(word);
        if(k){if(!variants.has(k))variants.set(k,[]);variants.get(k).push(word);}
      }
      literal.get(word).push(p.id);
    }
  }
  return terms=>{
    if(terms.length<2||!terms.some(t=>key(t)))return null;
    if(!literal)prepare();
    const resolved=terms.map(term=>{
      const postings=new Map(),spellings=new Map();
      const add=(word,cost)=>{for(const id of literal.get(word)||[])if(!postings.has(id)||cost<postings.get(id)){postings.set(id,cost);spellings.set(id,word);}};
      for(const form of forms(term)){
        add(form,0);const k=key(form);if(!k)continue;
        for(const word of variants.get(k)||[]){if(word===form)continue;const cost=edits(form,word);if(cost<=2)add(word,cost);}
      }
      return {term,postings,spellings};
    });
    let ids=[...resolved.reduce((a,b)=>a.postings.size<=b.postings.size?a:b).postings.keys()];
    ids=ids.filter(id=>resolved.every(r=>r.postings.has(id))&&resolved.some(r=>r.postings.get(id)>0));
    if(!ids.length)return null;
    return {ids,resolved};
  };
}
