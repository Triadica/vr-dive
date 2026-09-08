// Celestial Filament: twenty-four tilted glass orbit rings and four miniature satellites.
static float ogmField(float3 p,float time,thread float &material){
 float a=.34f+time*.045f,c=cos(a),s=sin(a);p.xy=float2(c*p.x-s*p.y,s*p.x+c*p.y);
 float best=length(p)-.125f;material=1;
 // Concentric rings share a plane. Only adjacent radial sectors can win the
 // distance query; this keeps all twelve fine rings without twelve torus tests.
 float tilt=.5f+.12f*sin(time*.04f);
 float ct=cos(tilt),st=sin(tilt);
 float3 q=float3(p.x,ct*p.y-st*p.z,st*p.y+ct*p.z);
 int nearest=int(clamp(round((length(q.xz)-.24f)/.0175f),0.0f,23.0f));
 for(int offset=-1;offset<=1;offset++){
  int j=clamp(nearest+offset,0,23);
  float k=float(j);
  float radius=.24f+k*.0175f,level=0.0f;
  float ring=length(float2(length(q.xz)-radius,q.y-level))-(.0028f+.0012f*float(j%3==0));
  if(ring<best){best=ring;material=float(j%3==0);}
  float pitch=6.2831853f/64.0f;
  float beadAngle=round(atan2(q.z,q.x)/pitch)*pitch;
  float chain=length(q-float3(radius*cos(beadAngle),level,radius*sin(beadAngle)))-.0045f;
  if(chain<best){best=chain;material=1;}
 }
 for(int j=0;j<12;j+=3){
   float k=float(j),radius=.24f+k*.035f,level=0.0f;
   float phase=time*(.10f+.006f*k)+k*2.4f;
   float bead=length(q-float3(radius*cos(phase),level,radius*sin(phase)))-.024f;
   if(bead<best){best=bead;material=1;}
 }
 return best;
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
