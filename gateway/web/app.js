'use strict';
const $=id=>document.getElementById(id);
let session=null,ws=null,eventWs=null,pc=null,mediaLocation=null,mic=null,screen=null,ctx=null,capture=null,heartbeat=null,screenTimer=null;
let generation=0,subtitle='',closing=false,audioMode=false,ready=false,pcmNodes=new Set(),fallbackEnd=0,mediaReady=false;
let mediaOrigin=0,pendingSubtitles=[],responseId=null,itemId=null,receivedMs=0,spans=[],fallbackSpans=[],restartTimer=null,mediaVersion=0;
const receiptState={base:0,played:0,generation:0};
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
async function iceComplete(peer){if(peer.iceGatheringState==='complete')return;await new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>reject(new Error('ICE 收集超时')),10000);
  peer.addEventListener('icegatheringstatechange',()=>{if(peer.iceGatheringState==='complete'){clearTimeout(timeout);resolve();}});
});}
async function closeMedia(){
  mediaVersion++;
  mediaReady=false;pc?.close();pc=null;$('avatar').srcObject=null;$('avatar').parentElement.classList.remove('playing');
  if(mediaLocation){const path=mediaLocation;mediaLocation=null;await api(path,{method:'DELETE'}).catch(()=>{});}
}
async function openMedia(){
  await closeMedia();if(!session||closing)return;
  const version=mediaVersion;
  const peer=pc=new RTCPeerConnection({iceServers:[]}),stream=new MediaStream();
  peer.addTransceiver('audio',{direction:'recvonly'});peer.addTransceiver('video',{direction:'recvonly'});
  peer.ontrack=event=>{stream.addTrack(event.track);$('avatar').srcObject=stream;$('avatar').parentElement.classList.add('playing');$('avatar').play().catch(()=>log('请点击播放器播放'));};
  peer.onconnectionstatechange=()=>{if(peer!==pc)return;mediaReady=peer.connectionState==='connected';if(mediaReady)stopFallback();if(peer.connectionState==='failed'){log('媒体连接失败，继续原生语音和字幕');closeMedia();}};
  await peer.setLocalDescription(await peer.createOffer());await iceComplete(peer);
  if(version!==mediaVersion)return;
  const response=await api(session.media_url,{method:'POST',headers:{'Content-Type':'application/sdp'},body:peer.localDescription.sdp});
  if(version!==mediaVersion){await api(response.headers.get('Location'),{method:'DELETE'}).catch(()=>{});return;}
  mediaLocation=response.headers.get('Location');mediaOrigin=Number(response.headers.get('X-Avatar-Timeline-Origin-Ms'));
  await peer.setRemoteDescription({type:'answer',sdp:await response.text()});
}
function stopFallback(){for(const node of pcmNodes){try{node.stop();}catch{}}pcmNodes.clear();fallbackEnd=0;fallbackSpans=[];}
function restartMedia(){clearTimeout(restartTimer);closeMedia();restartTimer=setTimeout(()=>openMedia().catch(error=>log(error.message)),100);}
function pcmBase64(buffer){const bytes=new Uint8Array(buffer);let raw='';for(let i=0;i<bytes.length;i+=8192)raw+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(raw);}
function fallback(event){
  if(mediaReady||!ctx)return;
  const bytes=Uint8Array.from(atob(event.delta),c=>c.charCodeAt(0));const samples=new DataView(bytes.buffer);
  const rate=event.sample_rate_hz||24000,buffer=ctx.createBuffer(1,bytes.length/2,rate),out=buffer.getChannelData(0);
  for(let i=0;i<out.length;i++)out[i]=samples.getInt16(i*2,true)/32768;
  const node=ctx.createBufferSource();node.buffer=buffer;node.connect(ctx.destination);
  const start=Math.max(ctx.currentTime+.04,fallbackEnd);fallbackEnd=start+buffer.duration;
  fallbackSpans.push({start,end:fallbackEnd,response:event.response_id,offset:receivedMs-buffer.duration*1000});
  fallbackSpans=fallbackSpans.filter(span=>span.end>ctx.currentTime-2);
  pcmNodes.add(node);node.onended=()=>pcmNodes.delete(node);node.start(start);
}
async function attachAvatar(source){
  session=await json('/v1/avatar/sessions',{avatar_id:'ava',source});generation=session.generation;
  $('session').textContent=`Ava · ${session.id}`;
  eventWs=await socket(session.events_url);eventWs.onmessage=event=>{
    const message=JSON.parse(event.data);
    if(message.type==='avatar.cleared'){
      generation=message.generation;subtitle='';pendingSubtitles=[];spans=[];$('subtitles').textContent='';stopFallback();
      // Replace the receiver to discard any old generation already in its jitter buffer.
      restartMedia();
    }else if(message.type==='avatar.subtitle')pendingSubtitles.push(message);
    else if(message.type==='avatar.audio.queued'){spans.push(message);spans=spans.slice(-120);}
    else if(message.type==='avatar.failed'){log('渲染故障，继续原生语音和字幕');closeMedia();}
    else if(message.type==='avatar.closed'){log('Avatar 已关闭');closeMedia();}
    else if(message.type==='avatar.resync_required'){log('事件接收过慢，请重新连接');disconnect();}
  };
  eventWs.onclose=()=>{if(!closing)log('Avatar 事件连接已断开');};
  await openMedia();log('Avatar 已创建，等待音频');
}
async function startMic(){
  mic=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  await ctx.audioWorklet.addModule('/mic-worklet.js');
  capture=new AudioWorkletNode(ctx,'pcm-capture');ctx.createMediaStreamSource(mic).connect(capture);capture.connect(ctx.destination);
  capture.port.onmessage=event=>{if(!ready)return;const samples=new Int16Array(event.data);let sum=0;for(const value of samples)sum+=value*value;
    const packet={type:'input_audio_buffer.append',audio:pcmBase64(event.data),format:'pcm16',sample_rate_hz:16000,is_speech:Math.sqrt(sum/samples.length)>350};
    if(screen?.active){const video=$('screen-preview');if(video?.videoWidth){const canvas=document.createElement('canvas');canvas.width=640;canvas.height=Math.round(video.videoHeight*640/video.videoWidth);canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);packet.video_frames=[canvas.toDataURL('image/jpeg',.7)];}}
    send(packet);
  };
}
async function connect(event){
  event.preventDefault();$('connect').disabled=true;$('status').textContent='连接中';closing=false;audioMode=$('source').value==='audio';
  try{
    ctx=new AudioContext({sampleRate:16000});await ctx.resume();
    if(audioMode)await attachAvatar({type:'audio'});
    else{
      ws=await socket('/v1/realtime','&duplex=1&model=omni-default&avatar=1');
      ws.onmessage=async event=>{const message=JSON.parse(event.data);
        try{
          if(message.type==='gateway.session.created'){
            if(message.avatar_available)await attachAvatar({type:'realtime',session_id:message.source_session_id});
            else log('Avatar 暂不可用，继续原生双工');
          }else if(message.type==='session.created'){
            ready=true;
            log('原生双工已就绪');
          }else if(['response.audio.delta','response.output_audio.delta'].includes(message.type)){
            if(responseId!==message.response_id){responseId=message.response_id;itemId=message.item_id;receivedMs=0;}
            receivedMs+=atob(message.delta).length/2/(message.sample_rate_hz||24000)*1000;fallback(message);
          }else if(['response.audio_transcript.delta','response.output_audio_transcript.delta','response.output_text.delta'].includes(message.type)&&!mediaReady){subtitle+=message.delta;$('subtitles').textContent=subtitle.slice(-800);}
          else if(message.type==='error')log(message.error?.message||JSON.stringify(message));
          else if(message.type==='output_audio_buffer.cleared'){stopFallback();subtitle='';$('subtitles').textContent='';}
        }catch(error){log(error.message);}
      };
      ws.onclose=()=>{if(!closing){log('语音连接断开，请关闭后重新连接');$('status').textContent='连接断开';ready=false;}};
      send({type:'session.update',session:{modalities:['audio','text'],input_audio_format:'pcm16',output_audio_format:'pcm16',playback_commit_policy:'ack_only',extra_body:{auto_response:true,force_listen_count:0}}});
      await startMic();
    }
    $('status').textContent=audioMode?'独立音频会话':'双工会话';$('disconnect').disabled=false;$('interrupt').disabled=false;$('screen').disabled=audioMode;
    $('audio-file').disabled=!audioMode;$('tone').disabled=!audioMode;$('source').disabled=true;$('token').disabled=true;
    heartbeat=setInterval(()=>{send({type:'session.heartbeat'});if(session)api(`/v1/avatar/sessions/${session.id}`).catch(error=>log(error.message));},20000);
  }catch(error){log(error.message);await disconnect();$('status').textContent='连接失败';}
}
async function disconnect(){
  closing=true;ready=false;clearInterval(heartbeat);clearTimeout(restartTimer);stopFallback();
  capture?.disconnect();capture=null;mic?.getTracks().forEach(track=>track.stop());mic=null;screen?.getTracks().forEach(track=>track.stop());screen=null;
  send({type:'session.close'});ws?.close();ws=null;eventWs?.close();eventWs=null;
  await closeMedia();if(session)await api(`/v1/avatar/sessions/${session.id}`,{method:'DELETE'}).catch(error=>log(error.message));session=null;
  await ctx?.close().catch(()=>{});ctx=null;subtitle='';pendingSubtitles=[];$('subtitles').textContent='';$('session').textContent='尚无活动会话';
  $('connect').disabled=false;$('disconnect').disabled=true;for(const id of ['interrupt','screen','audio-file','tone'])$(id).disabled=true;$('source').disabled=false;$('token').disabled=false;$('status').textContent='未连接';
}
async function interrupt(){
  stopFallback();subtitle='';pendingSubtitles=[];$('subtitles').textContent='';
  if(session)await closeMedia();
  if(audioMode&&session){await json(session.audio_url,{type:'clear'});}
  else{send({type:'response.cancel'});send({type:'output_audio_buffer.clear'});}
  log('已请求打断');
}
async function pushAudio(samples){
  if(!session||!audioMode)return;const expected=session.id,gen=generation;
  for(let offset=0;offset<samples.length;offset+=24000){
    if(!session||session.id!==expected||generation!==gen)break;
    const pcm=new Int16Array(Math.min(24000,samples.length-offset));for(let i=0;i<pcm.length;i++)pcm[i]=Math.round(Math.max(-1,Math.min(1,samples[offset+i]))*32767);
    await json(session.audio_url,{audio:pcmBase64(pcm.buffer),generation:gen,format:'pcm16',sample_rate_hz:24000});
    if(offset+24000<samples.length)await new Promise(resolve=>setTimeout(resolve,900));
  }
}
$('connect-form').addEventListener('submit',connect);$('disconnect').onclick=()=>disconnect();$('interrupt').onclick=()=>interrupt().catch(error=>log(error.message));
$('tone').onclick=()=>{const samples=new Float32Array(24000);for(let i=0;i<samples.length;i++)samples[i]=.16*Math.sin(2*Math.PI*440*i/24000);pushAudio(samples).catch(error=>log(error.message));};
$('audio-file').onchange=async event=>{try{const data=await ctx.decodeAudioData(await event.target.files[0].arrayBuffer());const offline=new OfflineAudioContext(1,Math.ceil(data.duration*24000),24000),node=offline.createBufferSource();node.buffer=data;node.connect(offline.destination);node.start();await pushAudio((await offline.startRendering()).getChannelData(0));}catch(error){log(error.message);}};
$('screen').onclick=async()=>{try{if(screen){screen.getTracks().forEach(track=>track.stop());screen=null;$('screen').textContent='共享屏幕';return;}screen=await navigator.mediaDevices.getDisplayMedia({video:true});let preview=$('screen-preview');if(!preview){preview=document.createElement('video');preview.id='screen-preview';preview.hidden=true;preview.muted=true;document.body.append(preview);}preview.srcObject=screen;await preview.play();$('screen').textContent='停止共享';screen.getVideoTracks()[0].onended=()=>{screen=null;$('screen').textContent='共享屏幕';};}catch(error){log(error.message);}};
function presented(now,metadata){
  if(session&&mediaReady){
    const mediaMs=metadata.mediaTime*1000;
    if(receiptState.generation!==generation){receiptState.base=mediaMs;receiptState.generation=generation;}
    receiptState.played=Math.max(0,mediaMs-receiptState.base);
    while(pendingSubtitles.length&&pendingSubtitles[0].generation===generation&&pendingSubtitles[0].start_ms<=mediaOrigin+mediaMs){subtitle+=pendingSubtitles.shift().text;$('subtitles').textContent=subtitle.slice(-800);}
  }
  $('avatar').requestVideoFrameCallback(presented);
}
if($('avatar').requestVideoFrameCallback)$('avatar').requestVideoFrameCallback(presented);
setInterval(()=>{
  const video=$('avatar');let timeline=null,played=null;
  if(mediaReady&&!video.paused&&!video.muted&&video.volume>0&&video.readyState>=3){
    timeline=mediaOrigin+video.currentTime*1000;
    const span=spans.findLast(s=>s.generation===generation&&s.start_ms<=timeline&&s.response_id===responseId);
    if(span)played=span.source_start_ms+Math.min(timeline,span.end_ms)-span.start_ms;
  }else if(ctx?.state==='running'&&!mediaReady){
    const output=ctx.getOutputTimestamp?.().contextTime??ctx.currentTime;
    const span=fallbackSpans.findLast(s=>s.response===responseId&&s.start<=output);
    if(span)played=span.offset+(Math.min(output,span.end)-span.start)*1000;
  }
  if(played!==null&&responseId&&played>=0)send({type:'playback.ack',response_id:responseId,item_id:itemId,played_ms:Math.floor(Math.min(played,receivedMs)),committed_ms:Math.floor(Math.min(played,receivedMs))});
  if(timeline!==null&&eventWs?.readyState===WebSocket.OPEN)eventWs.send(JSON.stringify({type:'playback.ack',generation,played_ms:Math.floor(timeline)}));
},250);
window.addEventListener('pagehide',()=>{send({type:'session.close'});ws?.close();eventWs?.close();mic?.getTracks().forEach(track=>track.stop());pc?.close();});
