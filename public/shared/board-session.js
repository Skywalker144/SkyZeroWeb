(function (scope) {
    class BoardGameSession {
        constructor({game,version,getHuman,positionOptions=()=>({}),onManifest=()=>{},onReady=()=>{},onPositionRequested=()=>{},render,analysis}) {
            Object.assign(this,{game,version,getHuman,positionOptions,onManifest,onReady,onPositionRequested,analysis});
            this.render=()=>{render();analysis.loading(!this.ready && !this.error,document.getElementById('status').textContent);};
            this.actions=[];
            this.state=null;
            this.lastSearch=null;
            this.ready=this.busy=this.thinking=false;
            this.error='';
            this.requestId=this.activeId=0;
        }
        syncPosition() {
            this.onPositionRequested();
            this.thinking=false;
            this.busy=this.ready;
            if(this.ready) {
                this.worker.postMessage({type:'cancel'});
                this.activeId=++this.requestId;
                this.worker.postMessage({id:this.activeId,type:'position',actions:[...this.actions],...this.positionOptions()});
            }
            this.render();
        }
        newGame() {
            this.actions=[];this.state=null;this.lastSearch=null;this.analysis.reset();
            if(this.error){this.boot();return;}
            this.syncPosition();
        }
        undo() {
            if(!this.ready || this.actions.length<=(this.getHuman()===-1?1:0))return;
            const remove=(this.actions.length%2?-1:1)===this.getHuman()?2:1;
            this.actions.splice(-remove);this.lastSearch=null;this.error='';
            this.analysis.trim(this.actions.length);this.syncPosition();
        }
        async boot() {
            this.worker?.terminate();this.ready=this.busy=this.thinking=false;this.error='';
            this.activeId=++this.requestId;
            const bootId=this.activeId;
            this.render();
            try {
                const response=await fetch(`/${this.game}/models/manifest.json?v=${this.version}`);
                if(!response.ok)throw new Error(`Manifest HTTP ${response.status}`);
                const manifest=await response.json();
                if(bootId!==this.activeId)return;
                this.manifest=manifest;this.onManifest(manifest);
                this.worker=new Worker(`/${this.game}/worker.js?v=${this.version}`);
                this.worker.onerror=event=>{this.error=event.message;this.busy=this.thinking=false;this.render();};
                this.worker.onmessage=({data})=> {
                    if(data.id!==this.activeId)return;
                    if(data.type==='ready'){this.onReady(data);this.ready=true;this.syncPosition();}
                    else if(data.type==='position') {
                        this.state=data.state;this.busy=false;
                        if(!this.state.finished && this.state.player!==this.getHuman()) {
                            this.thinking=true;this.lastSearch=null;this.activeId=++this.requestId;
                            this.worker.postMessage({id:this.activeId,type:'search',actions:[...this.actions],...this.positionOptions(),thinkMs:Number(document.getElementById('budget').value)});
                        }
                        this.render();
                    } else if(data.type==='progress'){this.lastSearch=data.result;this.render();}
                    else if(data.type==='move') {
                        this.lastSearch=data.result;this.analysis.record(this.actions.length+1,this.lastSearch);
                        this.actions.push(data.result.action);this.syncPosition();
                    } else if(data.type==='error'){this.error=data.message;this.busy=this.thinking=false;this.render();}
                };
                this.worker.postMessage({id:this.activeId,type:'init',manifest});
            } catch(problem) {if(bootId===this.activeId){this.error=problem.message;this.render();}}
        }
    }
    scope.BoardGameSession=BoardGameSession;
})(typeof self!=='undefined'?self:globalThis);
