// Petal Flame Marble — a lampworked flower whose thin petals curl through the globe.

static float pfmField(float3 p,float time,thread float &petal,thread float &vein){
    float a=time*.055f,c=cos(a),s=sin(a);p.xz=float2(c*p.x-s*p.z,s*p.x+c*p.z);
    float y=p.y+.08f,r=length(p.xz),ang=atan2(p.z,p.x);
    float twist=ang+1.10f*r+.38f*y;
    float lobes=cos(3.0f*twist);
    // Six broad, upward-curled sheets.  The angular wedge widens through the
    // middle then pinches at each tip, like a lampworked flower rather than a cage.
    float petalY=-.34f+.96f*r-.67f*r*r+.035f*cos(6.0f*twist);
    float width=.026f+.105f*smoothstep(.04f,.28f,r)*(1.0f-smoothstep(.48f,.68f,r));
    float outer=max(abs(y-petalY)-.018f,max(abs(sin(3.0f*twist))*r-width,r-.68f));
    float innerTwist=twist+.49f;
    float innerY=-.24f+.83f*r-.52f*r*r;
    float innerWidth=.022f+.073f*smoothstep(.025f,.18f,r)*(1.0f-smoothstep(.32f,.46f,r));
    float inner=max(abs(y-innerY)-.015f,max(abs(sin(3.0f*innerTwist))*r-innerWidth,r-.46f));
    // A third staggered whorl and branching raised ribs give visible relief.
    float thirdTwist=twist+.94f,thirdY=-.13f+.78f*r;
    float third=max(abs(y-thirdY)-.009f,
      max(abs(sin(3.0f*thirdTwist))*r-.040f,r-.32f));
    float rib=length(float2(y-petalY-.018f,
      sin(12.0f*twist+6.0f*r)*r/12.0f))-.0035f;
    rib=max(rib,max(r-.62f,abs(sin(3.0f*twist))*r-width));
    float petals=min(min(outer,inner),min(third,rib))*.40f;
    float stem=length(float2(r-.045f,max(abs(y+.50f)-.22f,0.0f)))-.028f;
    float pearl=length(p-float3(0,.28f,0))-.075f;
    petal=.5f+.5f*lobes;vein=abs(sin(3.0f*twist));
    return min(min(petals,stem),pearl);
}
static float pfmDE(float3 p,float t){float a,b;return pfmField(p,t,a,b);}
static float3 pfmNormal(float3 p,float t){const float e=.00045f;return normalize(float3(
 pfmDE(p+float3(e,0,0),t)-pfmDE(p-float3(e,0,0),t),
 pfmDE(p+float3(0,e,0),t)-pfmDE(p-float3(0,e,0),t),
 pfmDE(p+float3(0,0,e),t)-pfmDE(p-float3(0,0,e),t)));}

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
 float3 ro=(cam-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-cam),nn;
 if(!all(abs(ro)<DB_BOXDIMS-.001f)&&db_boxHit(ro,rd,DB_BOXDIMS,nn,true)<0)return float4(.010,.003,.009,1);
 ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize((u.patternTransform*float4(rd,0)).xyz);
 const float R=.84f;float b=dot(ro,rd),h=b*b-dot(ro,ro)+R*R;if(h<0)return float4(.010,.003,.009,1);
 float root=sqrt(h),entry=max(0.0f,-b-root);float3 sp=ro+rd*entry,sn=normalize(sp);
 float face=max(dot(-rd,sn),0.0f),fres=.038f+.962f*pow(1.0f-face,5.0f);
 float3 rr=refract(rd,sn,1.0f/1.47f);if(dot(ro,ro)<.84f*.84f||dot(rr,rr)<1e-8f)rr=rd;rr=normalize(rr);float3 rp=sp+rr*.004f;
 float rb=dot(rp,rr),rh=max(0.0f,rb*rb-dot(rp,rp)+R*R),end=-rb+sqrt(rh),z=0.0f;
 float3 col=float3(.026f,.005f,.020f)+fres*float3(.52f,.20f,.55f)*.35f;
 for(int i=0;i<288&&z<end;i++){float3 p=rp+rr*z;float petal,vein,d=pfmField(p,u.time,petal,vein);
  float glow=exp(-90.0f*max(d,0.0f));col+=float3(.42f,.025f,.10f)*glow*.014f;
  if(d<.00022f){float3 n=pfmNormal(p,u.time),l=normalize(float3(-.52f,.77f,.43f));float dif=max(dot(n,l),0.0f);
   float3 rose=float3(.92f,.035f,.18f),amber=float3(1.0f,.47f,.055f),violet=float3(.32f,.035f,.50f);
   float3 base=mix(rose,amber,petal);base=mix(base,violet,pow(1.0f-vein,7.0f)*.48f);
   if(length(p-float3(0,.28f,0))<.09f)base=float3(.85f,.95f,1.0f);
   float veinLine=pow(.5f+.5f*cos(atan2(p.z,p.x)*42.0f+length(p.xz)*14.0f),12.0f);
   base=mix(base,float3(.98f,.80f,.53f),veinLine*.4f);
   float spec=pow(max(dot(n,normalize(l-rr)),0.0f),68.0f);col+=base*(.24f+1.0f*dif)+spec*float3(1,.84,.72);break;}
  z+=clamp(d*.48f,.00012f,.022f);
 }
 float glint=pow(max(dot(reflect(rd,sn),normalize(float3(-.52f,.77f,.43f))),0.0f),132.0f);
 col+=fres*float3(.72f,.72f,1.0f)*.55f+glint*float3(1,.93,.82);

 // Exit refraction and Beer-Lambert absorption supply depth even between ribbons.
 // A single transmitted path is used; internal multiple reflections are approximated.
 float3 exitP=rp+rr*end,exitN=exitP/max(length(exitP),1e-6f);
 float3 outRay=refract(rr,-exitN,1.47f);
 if(dot(outRay,outRay)<1e-8f)outRay=reflect(rr,-exitN);
 float3 attenuation=exp(-float3(.18,.065,.04)*end);
 col=col*attenuation+marbleStudio(normalize(outRay))*.12f*attenuation*(1-fres);
 col+=marbleStudio(reflect(rd,sn))*(.025f+fres*.35f);
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
