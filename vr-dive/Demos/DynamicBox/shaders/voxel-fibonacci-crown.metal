// voxel-fibonacci-crown: independent runtime shader; mathematical occupancy rendered with 3D DDA.
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
// Golden-angle phyllotaxis: 89 oriented scales on a prolate envelope.
// Centers and radial directions are precomputed once as literals, not per ray step.
static constant float4 CROWN_SCALES[89]={
 float4(0.18450814f,-0.65258427f,0.00000000f,0.00000000f),
 float4(-0.15502608f,-0.63775281f,0.14201658f,2.39996323f),
 float4(0.02033974f,-0.62292135f,-0.23176087f,4.79992646f),
 float4(0.15370554f,-0.60808989f,0.20048166f,7.19988969f),
 float4(-0.26655785f,-0.59325843f,-0.04715033f,9.59985292f),
 float4(0.24235123f,-0.57842697f,-0.15416402f,11.99981615f),
 float4(-0.07852400f,-0.56359551f,0.29210536f,14.39977938f),
 float4(-0.14593289f,-0.54876404f,-0.28098490f,16.79974261f),
 float4(0.30979488f,-0.53393258f,0.11313659f,19.19970584f),
 float4(-0.31626002f,-0.51910112f,0.13054754f,21.59966907f),
 float4(0.14992400f,-0.50426966f,-0.32037888f,23.99963230f),
 float4(0.10912297f,-0.48943820f,0.34790122f,26.39959553f),
 float4(-0.32434782f,-0.47460674f,-0.18796619f,28.79955876f),
 float4(0.37559372f,-0.45977528f,-0.08257323f,31.19952199f),
 float4(-0.22643718f,-0.44494382f,0.32208359f,33.59948522f),
 float4(-0.05170871f,-0.43011236f,-0.39903254f,35.99944845f),
 float4(0.31393135f,-0.41528090f,0.26458168f,38.39941168f),
 float4(-0.41794702f,-0.40044944f,0.01728342f,40.79937491f),
 float4(0.30170411f,-0.38561798f,-0.30023589f,43.19933814f),
 float4(-0.01998131f,-0.37078652f,0.43211424f,45.59930136f),
 float4(-0.28136065f,-0.35595506f,-0.33716410f,47.99926459f),
 float4(0.44136695f,-0.34112360f,0.05938530f,50.39922782f),
 float4(-0.37037526f,-0.32629213f,0.25769746f,52.79919105f),
 float4(0.10024479f,-0.31146067f,-0.44559812f,55.19915428f),
 float4(0.22967069f,-0.29662921f,0.40080598f,57.59911751f),
 float4(-0.44476357f,-0.28179775f,-0.14189171f,59.99908074f),
 float4(0.42798110f,-0.26696629f,-0.19773812f,62.39904397f),
 float4(-0.18367542f,-0.25213483f,0.43888301f,64.79900720f),
 float4(-0.16238946f,-0.23730337f,-0.45148407f,67.19897043f),
 float4(0.42804004f,-0.22247191f,0.22496585f,69.59893366f),
 float4(-0.47095915f,-0.20764045f,0.12414332f,71.99889689f),
 float4(0.26515766f,-0.19280899f,-0.41238108f,74.39886012f),
 float4(0.08354313f,-0.17797753f,0.48611357f,76.79882335f),
 float4(-0.39211359f,-0.16314607f,-0.30367480f,79.19878658f),
 float4(0.49671925f,-0.14831461f,-0.04115217f,81.59874981f),
 float4(-0.33997539f,-0.13348315f,0.36750344f,83.99871304f),
 float4(0.00245192f,-0.11865169f,-0.50261420f,86.39867627f),
 float4(0.33887168f,-0.10382022f,0.37355657f,88.79863950f),
 float4(-0.50370344f,-0.08898876f,-0.04668312f,91.19860273f),
 float4(0.40395922f,-0.07415730f,-0.30659065f,93.59856596f),
 float4(-0.09095321f,-0.05932584f,0.49995927f,95.99852919f),
 float4(-0.27107957f,-0.04449438f,-0.43077254f,98.39849242f),
 float4(0.49142114f,-0.02966292f,0.13467816f,100.79845565f),
 float4(-0.45363816f,-0.01483146f,0.23279961f,103.19841888f),
 float4(0.17728458f,0.00000000f,-0.47819471f,105.59838211f),
 float4(0.19224859f,0.01483146f,0.47225387f,107.99834534f),
 float4(-0.46045052f,0.02966292f,-0.21821609f,110.39830857f),
 float4(0.48637696f,0.04449438f,-0.14995522f,112.79827180f),
 float4(-0.25693963f,0.05932584f,0.43842193f,115.19823503f),
 float4(-0.10647316f,0.07415730f,-0.49582693f,117.59819826f),
 float4(0.41240259f,0.08898876f,0.29295148f,119.99816149f),
 float4(-0.50048767f,0.10382022f,0.06237481f,122.39812472f),
 float4(0.32578307f,0.11865169f,-0.38274331f,124.79808795f),
 float4(0.01824499f,0.13348315f,0.50030907f,127.19805118f),
 float4(-0.34984853f,0.14831461f,-0.35500636f,129.59801441f),
 float4(0.49530806f,0.16314607f,0.02532543f,131.99797764f),
 float4(-0.38023890f,0.17797753f,0.31417231f,134.39794086f),
 float4(0.06774586f,0.19280899f,-0.48556899f,136.79790409f),
 float4(0.27621402f,0.20764045f,0.40114823f,139.19786732f),
 float4(-0.47124347f,0.22247191f,-0.10843202f,141.59783055f),
 float4(0.41745588f,0.23730337f,-0.23651381f,143.99779378f),
 float4(-0.14681187f,0.25213483f,0.45254970f,146.39775701f),
 float4(-0.19564796f,0.26696629f,-0.42894063f,148.79772024f),
 float4(0.42977126f,0.28179775f,0.18233090f,151.19768347f),
 float4(-0.43544107f,0.29662921f,0.15422429f,153.59764670f),
 float4(0.21445683f,0.31146067f,-0.40325546f,155.99760993f),
 float4(0.11287792f,0.32629213f,0.43685740f,158.39757316f),
 float4(-0.37341152f,0.34112360f,-0.24268341f,160.79753639f),
 float4(0.43315222f,0.35595506f,-0.07226758f,163.19749962f),
 float4(-0.26653282f,0.37078652f,0.34070842f,165.59746285f),
 float4(-0.03307306f,0.38561798f,-0.42435025f,167.99742608f),
 float4(0.30567305f,0.40044944f,0.28555632f,170.39738931f),
 float4(-0.41053662f,0.41528090f,-0.00400557f,172.79735254f),
 float4(0.29933198f,0.43011236f,-0.26888870f,175.19731577f),
 float4(-0.03824575f,0.44494382f,0.39185316f,177.59727900f),
 float4(-0.23099459f,0.45977528f,-0.30745809f,179.99724223f),
 float4(0.36849184f,0.47460674f,0.06889526f,182.39720546f),
 float4(-0.30953933f,0.48943820f,0.19268752f,184.79716869f),
 float4(0.09515635f,0.50426966f,-0.34068329f,187.19713192f),
 float4(0.15472703f,0.51910112f,0.30515997f,189.59709515f),
 float4(-0.30867632f,0.53393258f,-0.11615370f,191.99705838f),
 float4(0.29383204f,0.54876404f,-0.11794767f,194.39702161f),
 float4(-0.13086992f,0.56359551f,0.27269878f,196.79698484f),
 float4(-0.08328511f,0.57842697f,-0.27488953f,199.19694807f),
 float4(0.23287370f,0.59325843f,0.13800754f,201.59691130f),
 float4(-0.24724806f,0.60808989f,0.05183323f,203.99687453f),
 float4(0.13565534f,0.62292135f,-0.18900909f,206.39683776f),
 float4(0.02498288f,0.63775281f,0.20875261f,208.79680099f),
 float4(-0.13991715f,0.65258427f,-0.12027653f,211.19676422f),
};
static float2 voxelField(float3 p,float time){
 if(length(p)>.86f)return float2(0);
 int center=int(floor((p.y+.66f)*89.0f/1.32f-.5f));
 float best=2.0f,material=0;
 // Vertical support is <=.10; +/-8 indices covers every potentially occupied scale.
 for(int offset=-8;offset<=8;offset++){
  int i=center+offset;if(i<0||i>=89)continue;
  float4 scale=CROWN_SCALES[i];float3 delta=p-scale.xyz;
  float2 radial=normalize(scale.xz),tangent=float2(-radial.y,radial.x);
  float3 local=float3(dot(delta.xz,radial),delta.y,dot(delta.xz,tangent));
  // Tilt each scale tip outwards. Hollow plates produce overlapping open edges.
  local.x-=local.y*.28f;
  float shape=length(local/float3(.062f,.095f,.092f));
  float wall=abs(shape-.82f)*.062f-.009f;
  wall=max(wall,-local.x-.015f);
  if(wall<best){best=wall;material=fract(float(i)*.381966f);}
 }
 return float2(1.0f-smoothstep(0.0f,.009f,best),material);
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
                float3 base=mix(float3(.025f,.19f,.12f),float3(.22f,.53f,.24f),vein);
                float gold=smoothstep(.72f,.98f,sample.y);
                base=mix(base,float3(.88f,.59f,.15f),gold);
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
