import {describe,it,expect} from 'vitest';
import {verificationLabel,statusLabel} from './simulation';
describe('truthful task labels',()=>{
 it('does not label a successful navigation-only task as manipulation failure',()=>{expect(verificationLabel({verified:false,goals:[]},'completed')).toBe('无需操作校验');expect(verificationLabel({},'cancelled')).toBe('未测量');});
 it('distinguishes measured failures and successes',()=>{const goals=[{goal_id:'v',object:'valve_1',verified:false}];expect(verificationLabel({goals,verified:false},'failed')).toBe('未通过');expect(verificationLabel({goals,verified:true},'completed')).toBe('通过');expect(statusLabel('planned')).toBe('待执行');});
});
