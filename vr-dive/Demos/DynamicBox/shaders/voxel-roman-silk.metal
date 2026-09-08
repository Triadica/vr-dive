// voxel-roman-silk: independent runtime shader; mathematical occupancy rendered with 3D DDA.
static bool matterRay(DynamicBoxVertexOut in, constant DynamicBoxUniforms &u,
                      constant float4x4 *v2w, thread float3 &ro, thread float3 &rd) {
    uint vi=min(in.viewIndex,max(u.viewCount,1u)-1u);
    float3 eye=v2w[vi][3].xyz;
    ro=(eye-u.objectCenter.xyz)/max(u.boxScale,0.0001f);
    rd=normalize(in.worldPos-eye);
    if (!all(abs(ro)<DB_BOXDIMS-0.001f)) {
        float3 normal;
        if (db_boxHit(ro,rd,DB_BOXDIMS,normal,true)<0.0f) return false;
    }
    ro=(u.patternTransform*float4(ro,1)).xyz;
    rd=normalize((u.patternTransform*float4(rd,0)).xyz);
    return true;
}
static bool matterBound(float3 ro, float3 rd, float radius, thread float2 &span) {
    float b=dot(ro,rd), h=b*b-dot(ro,ro)+radius*radius;
    if (h<0.0f) return false;
    span=float2(max(0.0f,-b-sqrt(h)),-b+sqrt(h));
    return span.y>span.x;
}
static float3 matterShade(float3 normal, float3 rd, float3 base,
                          float occlusion, float translucency) {
    float3 n=dot(normal,rd)>0.0f ? -normal : normal;
    float3 light=normalize(float3(-0.55f,0.85f,0.65f));
    float diffuse=max(dot(n,light),0.0f);
    float back=pow(max(dot(-n,light),0.0f),2.0f)*translucency;
    float spec=pow(max(dot(n,normalize(light-rd)),0.0f),54.0f);
    float fresnel=pow(1.0f-max(dot(n,-rd),0.0f),4.0f);
    float3 color=base*(0.19f+0.95f*diffuse)*occlusion;
    color+=float3(1.0f,0.58f,0.30f)*back*0.32f;
    color+=float3(0.24f,0.52f,0.66f)*fresnel*0.25f;
    color+=float3(0.90f,0.92f,0.84f)*spec*0.5f;
    return color/(1.0f+0.18f*color);
}

static constant float VOXEL_SIZE=.016f;
// Roman Silk — thickened Steiner/Roman quartic, sampled into solid cubic voxels.
// Equation: x²y²+x²z²+y²z²-a xyz=0, a=1.30.
// Reference: https://mathworld.wolfram.com/RomanSurface.html (equation 13).
// Gradient normalization is only for occupancy thickness, never sphere tracing.
static float2 voxelField(float3 p,float time){
 if(length(p)>.77f||any(abs(p)>.655f))return float2(0);
 float3 q=p*p;
 float f=q.x*q.y+q.y*q.z+q.z*q.x-1.30f*p.x*p.y*p.z;
 float3 g=2.0f*p*(q.yzx+q.zxy)-1.30f*p.yzx*p.zxy;
 float distance=abs(f)/max(length(g),.02f);
 float thickness=.022f;
 float fill=1.0f-smoothstep(thickness,thickness+.012f,distance);
 return float2(fill,.5f+.5f*sin((p.x+p.y-p.z)*9.0f));
}

static bool voxelBox(float3 ro,float3 rd,float3 center,float halfSize,
                     thread float &distance,thread float3 &normal) {
    bool3 parallel=abs(rd)<0.0000001f;
    if (any(parallel && (abs(ro-center)>halfSize))) return false;
    float3 inverse=1.0f/select(rd,float3(0.0000001f),abs(rd)<0.0000001f);
    float3 a=(center-halfSize-ro)*inverse, b=(center+halfSize-ro)*inverse;
    float3 nearV=min(a,b), farV=max(a,b);
    nearV=select(nearV,float3(-1.0e20f),parallel);
    farV=select(farV,float3(1.0e20f),parallel);
    float nearT=max(nearV.x,max(nearV.y,nearV.z)), farT=min(farV.x,min(farV.y,farV.z));
    if (nearT>farT || farT<0.0f) return false;
    bool inside=nearT<0.0f;
    distance=inside?farT:nearT;
    float3 face=inside?farV:nearV;
    // Deterministic tie-breaking at voxel edges gives a stable face normal.
    int axis=inside?(face.x<=face.y && face.x<=face.z?0:(face.y<=face.z?1:2)):
                    (face.x>=face.y && face.x>=face.z?0:(face.y>=face.z?1:2));
    normal=float3(0); normal[axis]=(inside?1.0f:-1.0f)*sign(rd[axis]);
    return true;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    const float3 background=float3(0.003f,0.005f,0.010f);
    float3 ro,rd;
    if (!matterRay(in,u,v2w,ro,rd)) return float4(background,1);

    float2 span;
    // Rotate eye and ray together: the lattice stays rigid and occupancy does not flicker.
    float angle=u.time*.065f+.35f,c=cos(angle),s=sin(angle);
    ro.xz=float2(c*ro.x-s*ro.z,s*ro.x+c*ro.z);
    rd.xz=float2(c*rd.x-s*rd.z,s*rd.x+c*rd.z);
    if (!matterBound(ro,rd,.94f,span)) return float4(background,1);
    float travel=span.x+0.00001f;
    int3 cell=int3(floor((ro+rd*travel)/VOXEL_SIZE));
    int3 direction=int3(sign(rd));
    float3 safeAbs=max(abs(rd),float3(0.0000001f));
    float3 delta=VOXEL_SIZE/safeAbs;
    float3 cellBoundary=(float3(cell)+select(float3(0),float3(1),rd>0.0f))*VOXEL_SIZE;
    float3 next=(cellBoundary-ro)/select(rd,float3(0.0000001f),abs(rd)<0.0000001f);
    next=select(next,float3(1.0e20f),abs(rd)<0.0000001f);
    // 224 > 2*.94/.016*sqrt(3) + boundary cells; enough even along a diagonal.
    for (int step=0; step<224 && travel<span.y; ++step) {
        float exitT=min(next.x,min(next.y,next.z));
        float3 center=(float3(cell)+0.5f)*VOXEL_SIZE;
        float2 sample=voxelField(center,u.time);
        if (sample.x>0.025f) {
            float halfSize=VOXEL_SIZE*0.485f*sqrt(sample.x);
            float hitT; float3 n;
            if (voxelBox(ro,rd,center,halfSize,hitT,n) && hitT>=travel-0.0001f && hitT<=exitT+0.0001f) {
                float3 p=ro+rd*hitT;
                float3 local=(p-center)/max(halfSize,0.0001f);
                // Mild bevel shading; occupancy and silhouette remain exact boxes.
                float3 shadeNormal=normalize(n+0.12f*local);
                float occ=0.0f;
                occ+=voxelField(center+n*VOXEL_SIZE,u.time).x;
                occ+=voxelField(center+n*VOXEL_SIZE+float3(0,VOXEL_SIZE,0),u.time).x*0.5f;
                occ=clamp(1.0f-occ*0.3f,0.5f,1.0f);
                float vein=sample.y;
                float3 base=mix(float3(.18f,.045f,.32f),float3(.55f,.30f,.74f),vein);
                float gold=smoothstep(.72f,.98f,sample.y);
                base=mix(base,float3(.97f,.68f,.32f),gold);
                base*=0.70f+0.30f*sample.x;
                return float4(matterShade(shadeNormal,rd,base,occ,0.0f),1);
            }
        }
        // Advance every tied axis. Parallel rays never step their zero axis.
        bool3 crossed=next<=exitT+0.000001f;
        cell+=select(int3(0),direction,crossed);
        next+=select(float3(0),delta,crossed);
        travel=exitT;
    }
    return float4(background,1);
}
