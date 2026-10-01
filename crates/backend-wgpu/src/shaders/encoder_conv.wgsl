struct Params { p:vec4<u32> };
@group(0) @binding(0) var<uniform> pc:Params;
@group(0) @binding(1) var<storage,read> projection:array<f32>;
@group(0) @binding(2) var<storage,read> kernel:array<f32>;
@group(0) @binding(3) var<storage,read> segments:array<u32>;
@group(0) @binding(4) var<storage,read_write> dst:array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) nw:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>) {
    let i=flat_wg(wid,nw)*256u+lid.x;
    let tokens=pc.p.x;
    let hidden=pc.p.y;
    let width=pc.p.z;
    if(i>=tokens*hidden) { return; }
    let token=i/hidden;
    let channel=i%hidden;
    let segment=segments[token];
    if(segment==0u) { dst[i]=0.0; return; }
    var sum=0.0;
    for(var tap=0u;tap<width;tap+=1u) {
        let source=i32(token)+i32(tap)-i32(width/2u);
        if(source>=0 && source<i32(tokens)) {
            let t=u32(source);
            if(segments[t]==segment) {
                let base=t*3u*hidden+channel;
                sum+=kernel[channel*width+tap]*(projection[base]*projection[base+2u*hidden]);
            }
        }
    }
    dst[i]=projection[token*3u*hidden+hidden+channel]*sum;
}
