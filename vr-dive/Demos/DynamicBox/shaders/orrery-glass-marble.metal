// Orrery Glass Marble — a tiny orbital instrument suspended inside optical glass.

static float ogmTorus(float3 p,float2 radii){return length(float2(length(p.xz)-radii.x,p.y))-radii.y;}
static float ogmField(float3 p,float time,thread float &material){
    float a=time*.08f,c=cos(a),s=sin(a);p.xz=float2(c*p.x-s*p.z,s*p.x+c*p.z);
    float3 q=p;float d0=ogmTorus(q,float2(.48f,.014f));
    q=float3(p.x,p.z,p.y);float d1=ogmTorus(q,float2(.37f,.012f));
    float ca=cos(.68f),sa=sin(.68f);q=float3(ca*p.x-sa*p.y,sa*p.x+ca*p.y,p.z);
    float d2=ogmTorus(q,float2(.57f,.010f));
    float3 planetA=float3(.48f*cos(time*.13f),.02f,.48f*sin(time*.13f));
    float3 planetB=float3(.30f*cos(-time*.10f+2.1f),.19f,.30f*sin(-time*.10f+2.1f));
    float sun=length(p)-.105f,pa=length(p-planetA)-.052f,pb=length(p-planetB)-.037f;
    // Satellites have paired miniature rings and the main orbit carries pearl markers.
    float3 satellite=p-planetA;
    float satelliteRing=ogmTorus(satellite,float2(.085f,.0035f));
    satelliteRing=min(satelliteRing,ogmTorus(satellite,float2(.10f,.0025f)));
    float pitch=6.2831853f/48.0f;
    float theta=round(atan2(p.z,p.x)/pitch)*pitch;
    float pearls=length(p-float3(.48f*cos(theta),0,.48f*sin(theta)))-.007f;
    float ring=min(min(d0,min(d1,d2)),min(satelliteRing,pearls)),planet=min(sun,min(pa,pb));material=step(planet,ring);
    return min(ring,planet);
}
static float ogmDE(float3 p,float t){float m;return ogmField(p,t,m);}
static float3 ogmNormal(float3 p,float t){const float e=.00045f;return normalize(float3(
 ogmDE(p+float3(e,0,0),t)-ogmDE(p-float3(e,0,0),t),
 ogmDE(p+float3(0,e,0),t)-ogmDE(p-float3(0,e,0),t),
 ogmDE(p+float3(0,0,e),t)-ogmDE(p-float3(0,0,e),t)));}

// Procedural studio reflection, evaluated in object space for stable stereo views.
static float3 marbleStudio(float3 d){
 float key=pow(max(dot(d,normalize(float3(-.55,.75,1))),0.0f),42.0f);
 float strip=exp(-pow((d.x-.55f)/.11f,2.0f)-pow((d.y-.22f)/.48f,4.0f));
 return mix(float3(.02,.03,.05),float3(.19,.24,.29),smoothstep(-.5f,.85f,d.y))
 +float3(1,.94,.83)*key*2.2f+float3(.45,.72,1)*strip*.8f;
}

fragment float4 dynamicBoxFragment(
 DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms &u [[buffer(0)]],
 constant float4x4 *v2w [[buffer(1)]],constant float4x4 *vp [[buffer(2)]]){
 uint vi=min(in.viewIndex,u.viewCount-1u);float3 cam=v2w[vi][3].xyz;
 float3 ro=(cam-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-cam),nn;
 if(!all(abs(ro)<DB_BOXDIMS-.001f)&&db_boxHit(ro,rd,DB_BOXDIMS,nn,true)<0)return float4(.002,.004,.011,1);
 ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize((u.patternTransform*float4(rd,0)).xyz);
 const float R=.85f;float b=dot(ro,rd),h=b*b-dot(ro,ro)+R*R;if(h<0)return float4(.002,.004,.011,1);
 float root=sqrt(h),entry=max(0.0f,-b-root);float3 sp=ro+rd*entry,sn=normalize(sp);
 float face=max(dot(-rd,sn),0.0f),fres=.035f+.965f*pow(1.0f-face,5.0f);
 float3 rr=refract(rd,sn,1.0f/1.47f);if(dot(ro,ro)<.84f*.84f||dot(rr,rr)<1e-8f)rr=rd;rr=normalize(rr);float3 rp=sp+rr*.004f;
 float rb=dot(rp,rr),rh=max(0.0f,rb*rb-dot(rp,rp)+R*R),end=-rb+sqrt(rh),z=0.0f;
 float3 col=float3(.004f,.009f,.024f)+fres*float3(.22f,.48f,.88f)*.42f;
 // Additive near-misses make hair-thin rings readable without inflating their geometry.
 for(int i=0;i<288&&z<end;i++){float3 p=rp+rr*z;float mat,d=ogmField(p,u.time,mat);
  float glow=exp(-110.0f*max(d,0.0f));col+=mix(float3(.02f,.22f,.50f),float3(.65f,.24f,.035f),mat)*glow*.018f;
  if(d<.00022f){float3 n=ogmNormal(p,u.time),l=normalize(float3(-.48f,.81f,.34f));float dif=max(dot(n,l),0.0f);
   float radial=length(p);float3 metal=mix(float3(.10f,.52f,.92f),float3(1.0f,.39f,.06f),mat);
   if(mat>.5f&&radial<.15f)metal=float3(1.0f,.72f,.12f);
   float engraving=.5f+.5f*sin(p.y*145.0f+p.x*9.0f);
   if(radial<.15f)metal=mix(metal,float3(.12f,.63f,.55f),engraving*.55f);
   float spec=pow(max(dot(n,normalize(l-rr)),0.0f),80.0f);col+=metal*(.35f+1.15f*dif)+spec;break;}
  z+=clamp(d*.48f,.00012f,.022f);
 }
 float glint=pow(max(dot(reflect(rd,sn),normalize(float3(-.48f,.81f,.34f))),0.0f),140.0f);
 col+=fres*float3(.58f,.82f,1.0f)*.55f+glint*float3(1,.93,.75);

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
