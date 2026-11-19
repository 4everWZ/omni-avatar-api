'use strict';
const $=id=>document.getElementById(id);
let ws=null,mic=null,screen=null,ctx=null,capture=null,heartbeat=null,connectionAttempt=0;
let subtitle='',closing=false,ready=false,pcmNodes=new Set(),playbackEnd=0;
let voiceEpoch=-1,awaitingClear=false,blockedResponses=new Set();
let responseId=null,itemId=null,receivedMs=0,playbackSpans=[];
function log(text){const rows=$('events').textContent.split('\n').filter(Boolean);rows.push(`${new Date().toLocaleTimeString()} ${text}`);$('events').textContent=rows.slice(-40).join('\n');$('events').scrollTop=$('events').scrollHeight;}
function send(event){if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(event));}
function auth(){return {Authorization:`Bearer ${$('token').value}`};}
async function api(path,options={}){
  const response=await fetch(path,{...options,headers:{...auth(),...options.headers}});
  if(!response.ok){let message=await response.text();try{message=JSON.parse(message).error?.message||message;}catch{}throw new Error(`${response.status}: ${message.slice(0,180)}`);}
  return response;
}
async function json(path,body){return (await api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();}
async function socket(path,query=''){
  const {ticket}=await json('/v1/browser-tickets',{path});
  const url=new URL(path,location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';url.search=`ticket=${encodeURIComponent(ticket)}${query}`;
  const socket=new WebSocket(url);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=()=>reject(new Error('无法连接会话，请检查凭据与服务状态'));});
  return socket;
}
function stopPlayback(){for(const node of pcmNodes){try{node.stop();}catch{}}pcmNodes.clear();playbackEnd=0;playbackSpans=[];}
function pcmBase64(buffer){const bytes=new Uint8Array(buffer);let raw='';for(let i=0;i<bytes.length;i+=8192)raw+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(raw);}
function playAudio(event){
  if(!ctx)return;
  const bytes=Uint8Array.from(atob(event.delta),c=>c.charCodeAt(0));const samples=new DataView(bytes.buffer);
  const rate=event.sample_rate_hz||24000,buffer=ctx.createBuffer(1,bytes.length/2,rate),out=buffer.getChannelData(0);
  for(let i=0;i<out.length;i++)out[i]=samples.getInt16(i*2,true)/32768;
  const node=ctx.createBufferSource();node.buffer=buffer;node.connect(ctx.destination);
  const start=Math.max(ctx.currentTime+.04,playbackEnd);playbackEnd=start+buffer.duration;
  playbackSpans.push({start,end:playbackEnd,response:event.response_id,offset:receivedMs-buffer.duration*1000});
  playbackSpans=playbackSpans.filter(span=>span.end>ctx.currentTime-2);
  pcmNodes.add(node);node.onended=()=>pcmNodes.delete(node);node.start(start);
}
async function startMic(){
  const attempt=connectionAttempt,context=ctx;
  const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  if(closing||attempt!==connectionAttempt){stream.getTracks().forEach(track=>track.stop());return;}
  mic=stream;
  await context.audioWorklet.addModule('/mic-worklet.js');
  if(closing||attempt!==connectionAttempt)return;
  capture=new AudioWorkletNode(ctx,'pcm-capture');ctx.createMediaStreamSource(mic).connect(capture);capture.connect(ctx.destination);
  capture.port.onmessage=event=>{if(!ready)return;
    const packet={type:'input_audio_buffer.append',audio:pcmBase64(event.data),format:'pcm16',sample_rate_hz:16000};
    if(screen?.active){const video=$('screen-preview');if(video?.videoWidth){const canvas=document.createElement('canvas');canvas.width=640;canvas.height=Math.round(video.videoHeight*640/video.videoWidth);canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);packet.video_frames=[canvas.toDataURL('image/jpeg',.7).split(',')[1]];}}
    send(packet);
  };
}
async function connect(event){
  event.preventDefault();$('connect').disabled=true;$('status').textContent='连接中';closing=false;
  const attempt=++connectionAttempt;$('disconnect').disabled=false;$('token').disabled=true;
  try{
    ctx=new AudioContext({sampleRate:16000});await ctx.resume();
    if(attempt!==connectionAttempt)return;
    const openedSocket=await socket('/v1/realtime','&duplex=1&model=omni-default');
    if(attempt!==connectionAttempt){openedSocket.close();return;}
    ws=openedSocket;
    const activeSocket=ws;
    ws.onmessage=async event=>{if(ws!==activeSocket||closing)return;const message=JSON.parse(event.data);
      try{
        const epoch=message.metadata?.epoch??message.epoch??message.session?.epoch;
        const output=['response.audio.delta','response.output_audio.delta','response.audio_transcript.delta','response.output_audio_transcript.delta','response.output_text.delta'].includes(message.type);
        if(output&&(awaitingClear||blockedResponses.has(message.response_id)||(Number.isInteger(epoch)&&epoch<voiceEpoch)))return;
        if(Number.isInteger(epoch)&&epoch>voiceEpoch){if(voiceEpoch>=0){stopPlayback();subtitle='';$('subtitles').textContent='';}voiceEpoch=epoch;}
        if(message.type==='session.created'){
          ready=true;
          log('原生双工已就绪');
        }else if(message.type==='response.created'||message.type==='response.speak'){
          const id=message.response_id??message.response?.id;
          if(awaitingClear){if(id)blockedResponses.add(id);return;}
          if(id&&responseId!==id){responseId=id;itemId=message.item_id??`item_${id}`;receivedMs=0;}
        }else if(['response.audio.delta','response.output_audio.delta'].includes(message.type)){
          if(responseId!==message.response_id){responseId=message.response_id;itemId=message.item_id;receivedMs=0;}
          receivedMs+=atob(message.delta).length/2/(message.sample_rate_hz||24000)*1000;playAudio(message);
        }else if(['response.audio_transcript.delta','response.output_audio_transcript.delta','response.output_text.delta'].includes(message.type)){subtitle+=message.delta;$('subtitles').textContent=subtitle.slice(-800);}
        else if(message.type==='error')log(message.error?.message||JSON.stringify(message));
        else if(message.type==='output_audio_buffer.cleared'){if(message.response_id??responseId)blockedResponses.add(message.response_id??responseId);awaitingClear=false;stopPlayback();subtitle='';$('subtitles').textContent='';}
      }catch(error){log(error.message);}
    };
    ws.onclose=()=>{if(ws===activeSocket&&!closing){log('语音连接断开，可重新连接');disconnect().finally(()=>$('status').textContent='连接断开');}};
    send({type:'session.update',session:{modalities:['audio','text'],input_audio_format:'pcm16',output_audio_format:'pcm16',playback_commit_policy:'ack_only',extra_body:{auto_response:true,force_listen_count:0}}});
    await startMic();
    if(attempt!==connectionAttempt)return;
    $('status').textContent='双工会话';$('disconnect').disabled=false;$('interrupt').disabled=false;$('screen').disabled=false;
    heartbeat=setInterval(()=>send({type:'session.heartbeat'}),20000);
  }catch(error){log(error.message);await disconnect();$('status').textContent='连接失败';}
}
async function disconnect(){
  connectionAttempt++;
  closing=true;ready=false;clearInterval(heartbeat);stopPlayback();
  capture?.disconnect();capture=null;mic?.getTracks().forEach(track=>track.stop());mic=null;screen?.getTracks().forEach(track=>track.stop());screen=null;
  send({type:'session.close'});ws?.close();ws=null;
  await ctx?.close().catch(()=>{});ctx=null;subtitle='';$('subtitles').textContent='';
  voiceEpoch=-1;awaitingClear=false;blockedResponses.clear();responseId=null;receivedMs=0;
  $('connect').disabled=false;$('disconnect').disabled=true;for(const id of ['interrupt','screen'])$(id).disabled=true;$('token').disabled=false;$('status').textContent='未连接';
}
async function interrupt(){
  stopPlayback();subtitle='';$('subtitles').textContent='';
  if(responseId)blockedResponses.add(responseId);
  awaitingClear=true;send({type:'response.cancel'});send({type:'output_audio_buffer.clear'});
  log('已请求打断');
}
$('connect-form').addEventListener('submit',connect);$('disconnect').onclick=()=>disconnect();$('interrupt').onclick=()=>interrupt().catch(error=>log(error.message));
$('screen').onclick=async()=>{try{if(screen){screen.getTracks().forEach(track=>track.stop());screen=null;$('screen').textContent='共享屏幕';return;}screen=await navigator.mediaDevices.getDisplayMedia({video:true});let preview=$('screen-preview');if(!preview){preview=document.createElement('video');preview.id='screen-preview';preview.hidden=true;preview.muted=true;document.body.append(preview);}preview.srcObject=screen;await preview.play();$('screen').textContent='停止共享';screen.getVideoTracks()[0].onended=()=>{screen=null;$('screen').textContent='共享屏幕';};}catch(error){log(error.message);}};
setInterval(()=>{
  let played=null;
  if(ctx?.state==='running'){
    const output=ctx.getOutputTimestamp?.().contextTime??ctx.currentTime;
    const span=playbackSpans.findLast(s=>s.response===responseId&&s.start<=output);
    if(span)played=span.offset+(Math.min(output,span.end)-span.start)*1000;
  }
  if(played!==null&&responseId&&played>=0)send({type:'playback.ack',response_id:responseId,item_id:itemId,played_ms:Math.floor(Math.min(played,receivedMs)),committed_ms:Math.floor(Math.min(played,receivedMs))});
},250);
window.addEventListener('pagehide',()=>{send({type:'session.close'});ws?.close();mic?.getTracks().forEach(track=>track.stop());screen?.getTracks().forEach(track=>track.stop());});
