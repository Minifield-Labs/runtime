
fn flat_wg(wid: vec3<u32>, numw: vec3<u32>) -> u32 {
    return wid.z * numw.y * numw.x + wid.y * numw.x + wid.x;
}
