// Agate Tide Marble — translucent glass enclosing fine, wind-folded mineral strata.
// The sphere is an optical boundary; only the internal laminae are ray-marched.

static float atmHash(float3 p) {
    p = fract(p * 0.1031f); p += dot(p, p.yzx + 33.33f);
    return fract((p.x + p.y) * p.z);
}
static float atmNoise(float3 p) {
    float3 i=floor(p), f=fract(p); f=f*f*(3.0f-2.0f*f);
    return mix(mix(mix(atmHash(i),atmHash(i+float3(1,0,0)),f.x),
                   mix(atmHash(i+float3(0,1,0)),atmHash(i+float3(1,1,0)),f.x),f.y),
               mix(mix(atmHash(i+float3(0,0,1)),atmHash(i+float3(1,0,1)),f.x),
                   mix(atmHash(i+float3(0,1,1)),atmHash(i+float3(1,1,1)),f.x),f.y),f.z);
}
static float atmField(float3 p, float time, thread float &phaseOut) {
    float a=0.13f*sin(time*0.10f), c=cos(a), s=sin(a);
    p.xz=float2(c*p.x-s*p.z,s*p.x+c*p.z);
    float warp=0.13f*sin(p.x*5.2f+p.z*1.7f)+0.075f*sin(p.z*8.3f-p.x*2.4f);
    warp+=0.045f*(atmNoise(p*5.0f)-0.5f);
    float phase=(p.y+warp)*56.0f+2.3f*sin(atan2(p.z,p.x)*2.0f+length(p.xz)*7.0f);
    phaseOut=phase;
    // Thinner doubled strata with a subordinate ripple; preserve clear spacing.
    float micro=.18f*sin(p.x*32.0f+p.z*23.0f);
    float ribbons=abs(sin(phase+micro))*.011f-.0018f;
    float core=length(p/float3(0.71f,0.67f,0.71f))-1.0f;
    return max(ribbons,core*0.22f);
}
static float atmDE(float3 p,float t){float q;return atmField(p,t,q);}
static float3 atmNormal(float3 p,float t){const float e=.00045f;return normalize(float3(
    atmDE(p+float3(e,0,0),t)-atmDE(p-float3(e,0,0),t),
    atmDE(p+float3(0,e,0),t)-atmDE(p-float3(0,e,0),t),
    atmDE(p+float3(0,0,e),t)-atmDE(p-float3(0,0,e),t)));}
static float3 atmPalette(float phase,float3 p){
    float band=.5f+.5f*sin(phase*.5f);
    float fine=.5f+.5f*cos(phase*2.0f+p.x*9.0f);
    float3 deep=float3(.015f,.16f,.24f), aqua=float3(.05f,.64f,.72f), pearl=float3(.88f,.93f,.86f);
    return mix(mix(deep,aqua,band),pearl,pow(fine,8.0f)*.82f);
}

// Procedural studio reflection, evaluated in object space for stable stereo views.
static float3 marbleStudio(float3 d){
 float key=pow(max(dot(d,normalize(float3(-.55,.75,1))),0.0f),42.0f);
 float strip=exp(-pow((d.x-.55f)/.11f,2.0f)-pow((d.y-.22f)/.48f,4.0f));
 return mix(float3(.02,.03,.05),float3(.19,.24,.29),smoothstep(-.5f,.85f,d.y))
 +float3(1,.94,.83)*key*2.2f+float3(.45,.72,1)*strip*.8f;
}

static float4 marbleSample(
 DynamicBoxVertexOut in,constant DynamicBoxUniforms &u,
 constant float4x4 *v2w,constant float4x4 *vp){
 uint vi=min(in.viewIndex,u.viewCount-1u);float3 cam=v2w[vi][3].xyz;
 float3 ro=(cam-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-cam);
 float3 unused;if(!all(abs(ro)<DB_BOXDIMS-.001f)&&db_boxHit(ro,rd,DB_BOXDIMS,unused,true)<0)return float4(.004,.007,.012,1);
 ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize((u.patternTransform*float4(rd,0)).xyz);
 const float radius=.84f;float b=dot(ro,rd),h=b*b-dot(ro,ro)+radius*radius;
 if(h<0)return float4(.004,.007,.012,1);float root=sqrt(h),entry=max(0.0f,-b-root);
 float3 shellP=ro+rd*entry,shellN=normalize(shellP);float facing=max(dot(-rd,shellN),0.0f);
 float fres=.035f+.965f*pow(1.0f-facing,5.0f);float3 innerRD=refract(rd,shellN,1.0f/1.47f);
 if(dot(ro,ro)<.84f*.84f||dot(innerRD,innerRD)<1e-8f)innerRD=rd;innerRD=normalize(innerRD);float3 innerRO=shellP+innerRD*.004f;
 float ib=dot(innerRO,innerRD),ih=max(0.0f,ib*ib-dot(innerRO,innerRO)+radius*radius);float end=-ib+sqrt(ih),travel=0.0f;
 float3 col=float3(.014f,.035f,.052f)*(1.0f-fres)+float3(.25f,.55f,.66f)*fres*.34f;
 for(int i=0;i<288&&travel<end;i++){float3 p=innerRO+innerRD*travel;float phase,d=atmField(p,u.time,phase);
  if(d<.00022f){float3 n=atmNormal(p,u.time),l=normalize(float3(-.55f,.74f,.39f));
   float dif=max(dot(n,l),0.0f),spec=pow(max(dot(reflect(-l,n),-innerRD),0.0f),60.0f);
   col+=atmPalette(phase,p)*(.27f+.92f*dif)*(1.0f-fres*.35f)+float3(1,.92,.78)*spec*.7f;break;}
  travel+=clamp(d*.48f,.00012f,.022f);
 }
 float glint=pow(max(dot(reflect(rd,shellN),normalize(float3(-.55f,.74f,.39f))),0.0f),120.0f);
 col+=float3(.72f,.92f,1.0f)*fres*.58f+float3(1.0f,.94f,.82f)*glint;

 // Exit refraction and Beer-Lambert absorption supply depth even between ribbons.
 // A single transmitted path is used; internal multiple reflections are approximated.
 float3 exitP=innerRO+innerRD*end,exitN=exitP/max(length(exitP),1e-6f);
 float3 outRay=refract(innerRD,-exitN,1.47f);
 if(dot(outRay,outRay)<1e-8f)outRay=reflect(innerRD,-exitN);
 float3 attenuation=exp(-float3(.18,.065,.04)*end);
 col=col*attenuation+marbleStudio(normalize(outRay))*.12f*attenuation*(1-fres);
 col+=marbleStudio(reflect(rd,shellN))*(.025f+fres*.35f);
 return float4(col/(1.0f+col*.28f),1);
}

// Two spatial subpixel samples soften curved sheet edges without temporal history.
// Derivatives are taken before the divergent marches, as required for stable quads.
fragment float4 dynamicBoxFragment(
 DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms &u [[buffer(0)]],
 constant float4x4 *v2w [[buffer(1)]],constant float4x4 *vp [[buffer(2)]]){
 float3 offset=(dfdx(in.worldPos)+dfdy(in.worldPos))*.22f;
 DynamicBoxVertexOut a=in,b=in;a.worldPos-=offset;b.worldPos+=offset;
 return (marbleSample(a,u,v2w,vp)+marbleSample(b,u,v2w,vp))*.5f;
}
