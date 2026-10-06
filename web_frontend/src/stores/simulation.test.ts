// @vitest-environment jsdom
import {beforeEach,afterEach,describe,it,expect,vi} from 'vitest';
import {createPinia,setActivePinia} from 'pinia';
import {useSimulationStore} from './simulation';
import type {SimulationStatus} from '../types/simulation';

vi.mock('../api/http',()=>({fetchSimulationStatus:vi.fn(),postCameraCommand:vi.fn(),postSimulationControl:vi.fn(),setOverlay:vi.fn(),errorMessage:()=> '请求失败',fetchTaskDetail:vi.fn().mockResolvedValue({samples:[],plan:null})}));

class Socket {
 static OPEN=1;static CONNECTING=0;static instances:Socket[]=[];
 readyState=0;binaryType='';onopen:(()=>void)|null=null;onclose:(()=>void)|null=null;onerror:(()=>void)|null=null;onmessage:((event:{data:string})=>void)|null=null;
 constructor(_url:string){Socket.instances.push(this);}
 close(){this.readyState=3;this.onclose?.();}
 open(){this.readyState=1;this.onopen?.();}
 message(payload:SimulationStatus){this.onmessage?.({data:JSON.stringify({type:'status',payload})});}
}
function status(id='A',time=1):SimulationStatus{return {state:'running',sim_time:time,frame_index:1,resolution:[960,540],fps:20,camera:{},available_cameras:[],active_task:{id,instruction:'test',status:'running',message:'',progress:0,runner_state:'PHASE_NAV',phase_index:0,phase_count:1},telemetry:{t:time,phase:0,arms:{}}};}
beforeEach(()=>{vi.useFakeTimers();setActivePinia(createPinia());Socket.instances=[];vi.stubGlobal('WebSocket',Socket);});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();});
describe('persistent simulation connection',()=>{
 it('connects once and does not reconnect after intentional disconnect',()=>{const store=useSimulationStore();store.connect();store.connect();expect(Socket.instances).toHaveLength(1);Socket.instances[0]!.open();expect(store.isOnline).toBe(true);store.disconnect();vi.advanceTimersByTime(5000);expect(Socket.instances).toHaveLength(1);expect(store.connectionState).toBe('idle');});
 it('reconnects after accidental closure and ignores the obsolete connection',()=>{const store=useSimulationStore();store.connect();const old=Socket.instances[0]!;old.open();old.close();vi.advanceTimersByTime(1500);expect(Socket.instances).toHaveLength(2);const next=Socket.instances[1]!;next.open();next.message(status('new'));old.message(status('old'));expect(store.task?.id).toBe('new');expect(store.isOnline).toBe(true);store.disconnect();});
 it('deduplicates samples and clears old task data on reset',()=>{const store=useSimulationStore();store.connect();const socket=Socket.instances[0]!;socket.open();socket.message(status());socket.message(status());expect(store.samples).toHaveLength(1);socket.message(status('B',0));expect(store.samples).toHaveLength(1);expect(store.samples[0]!.t).toBe(0);socket.message({...status(),active_task:null,last_task:null,telemetry:null});expect(store.samples).toHaveLength(0);expect(store.detail).toBe(null);store.disconnect();});
});
