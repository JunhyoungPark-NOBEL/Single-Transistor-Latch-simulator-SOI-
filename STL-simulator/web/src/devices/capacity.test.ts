import { afterEach, describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { builtinDevices } from "./library";
import { MAX_USER_DEVICES, parseStoredDevices, useDeviceLib } from "./store";
const device = builtinDevices(BUILTIN_META)[0];
afterEach(() => useDeviceLib.setState({devices: []}));
describe("five-device capacity", () => {
  it("caps creation paths but permits updates and frees a slot after deletion", () => {
    useDeviceLib.setState({devices: []});
    for(let i=0;i<MAX_USER_DEVICES;i++) useDeviceLib.getState().add({...device,id:undefined,name:`Device ${i}`});
    expect(() => useDeviceLib.getState().add(device)).toThrow(/limit/);
    expect(() => useDeviceLib.getState().duplicate(device,"copy")).toThrow(/limit/);
    expect(useDeviceLib.getState().importMany([device])).toBe(0);
    const id=useDeviceLib.getState().devices[0].id;
    useDeviceLib.getState().update(id,{device:{...device.device,vg:-1.9}});
    expect(useDeviceLib.getState().devices[0].device.vg).toBe(-1.9);
    useDeviceLib.getState().remove(id);
    expect(useDeviceLib.getState().importMany([device,device])).toBe(1);
    expect(useDeviceLib.getState().devices).toHaveLength(5);
  });
  it("preserves existing libraries larger than five without deleting user data", () => {
    const older=Array.from({length:7},(_,i)=>({...device,builtin:undefined,id:`user-${i}`,name:`D${i}`}));
    expect(parseStoredDevices(JSON.stringify(older))).toHaveLength(7);
  });
});
