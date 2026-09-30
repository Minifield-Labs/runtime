struct Params { p: vec4<u32>, q: vec4<u32> };
@group(0) @binding(0) var<uniform> pc: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
@group(0) @binding(4) var<storage, read> segments: array<u32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
var<workgroup> reduction: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(num_workgroups) nw:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
    let flat=flat_wg(wid,nw);
    // Flattened grids can round up. This uniform guard precedes every buffer
    // access and barrier, so padded groups can't touch another score row.
    if(flat>=pc.q.z) { return; }
    let heads=pc.p.x;
    let dim=pc.p.y;
    let tokens=pc.p.z;
    let group_size=pc.p.w;
    let lane=lid.x;
    let token=pc.q.x+flat/heads;
    let head=flat%heads;
    let kv_head=head/group_size;
    let kv_width=(heads/group_size)*dim;
    let qbase=token*heads*dim+head*dim;
    let sbase=flat*tokens;
    let query_segment=segments[token];
    let scale=bitcast<f32>(pc.q.y);
    for(var i=lane;i<tokens;i+=256u) {
        var score=-3.402823466e+38;
        if(query_segment!=0u && segments[i]==query_segment) {
            var dot=0.0;
            for(var d=0u;d<dim;d+=1u) { dot+=query[qbase+d]*key[i*kv_width+kv_head*dim+d]; }
            score=dot*scale;
        }
        scores[sbase+i]=score;
    }
    storageBarrier();
    var maximum=-3.402823466e+38;
    for(var i=lane;i<tokens;i+=256u) { maximum=max(maximum,scores[sbase+i]); }
    reduction[lane]=maximum;
    workgroupBarrier();
    for(var s=128u;s>0u;s>>=1u) { if(lane<s) { reduction[lane]=max(reduction[lane],reduction[lane+s]); } workgroupBarrier(); }
    maximum=reduction[0];
    workgroupBarrier();
    var denominator=0.0;
    for(var i=lane;i<tokens;i+=256u) {
        var e=0.0;
        if(query_segment!=0u && segments[i]==query_segment) { e=exp(scores[sbase+i]-maximum); }
        scores[sbase+i]=e;
        denominator+=e;
    }
    reduction[lane]=denominator;
    workgroupBarrier();
    for(var s=128u;s>0u;s>>=1u) { if(lane<s) { reduction[lane]+=reduction[lane+s]; } workgroupBarrier(); }
    denominator=reduction[0];
    storageBarrier();
    for(var d=lane;d<dim;d+=256u) {
        var sum=0.0;
        for(var i=0u;i<tokens;i+=1u) {
            if(query_segment!=0u && segments[i]==query_segment) {
                sum+=scores[sbase+i]*value[i*kv_width+kv_head*dim+d];
            }
        }
        if(query_segment==0u) { dst[qbase+d]=0.0; } else { dst[qbase+d]=sum/denominator; }
    }
}
