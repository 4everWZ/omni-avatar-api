class PcmCapture extends AudioWorkletProcessor {
  constructor(){super();this.samples=new Int16Array(16000);this.offset=0;}
  process(inputs){
    const input=inputs[0]?.[0];
    if(input)for(const value of input){
      this.samples[this.offset++]=Math.round(Math.max(-1,Math.min(1,value))*32767);
      if(this.offset===16000){this.port.postMessage(this.samples.buffer,[this.samples.buffer]);this.samples=new Int16Array(16000);this.offset=0;}
    }
    return true;
  }
}
registerProcessor('pcm-capture',PcmCapture);
