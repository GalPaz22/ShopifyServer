import {parentPort,workerData} from 'node:worker_threads';
import vm from 'node:vm';

// Runs one tenant's functions inside an empty VM context in this worker thread (memory-limited by the parent).
// The context has no require/import/process/timers/network and cannot compile strings; data arrives and leaves as JSON text.
const context=vm.createContext(Object.create(null),{codeGeneration:{strings:false,wasm:false}});
let loadError=null;
try{new vm.Script(workerData.source,{filename:'tenant-functions.js'}).runInContext(context,{timeout:1000});}catch(e){loadError=String(e?.message||e);}
const invoke=new vm.Script('__out=__run(__name,__input)');
parentPort.on('message',({id,name,input,timeout})=>{
 if(loadError)return parentPort.postMessage({id,error:'טעינת הפונקציות נכשלה: '+loadError});
 try{context.__name=name;context.__input=input;context.__out=null;invoke.runInContext(context,{timeout});parentPort.postMessage({id,out:context.__out});}
 catch(e){parentPort.postMessage({id,error:String(e?.message||e).slice(0,500)});}
 finally{context.__input=null;context.__out=null;}
});
