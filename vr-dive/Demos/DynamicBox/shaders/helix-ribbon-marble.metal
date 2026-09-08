// Helix Ribbon Marble — two hand-pulled glass ribbons twisting around a clear core.

static float hrmField(float3 p,float time,thread float &which,thread float &edge){
    float a=time*.07f,c=cos(a),s=sin(a);p.xy=float2(c*p.x-s*p.y,s*p.x+c*p.y);
    float r=length(p.xz),ang=atan2(p.z,p.x),twist=ang-p.y*5.4f;
    float radial=abs(r-(.36f+.055f*sin(p.y*8.0f)))-.085f;
    float ribbonA=abs(sin(twist))*(r+.08f)-.030f;
    float ribbonB=abs(sin(twist+1.5708f))*(r+.08f)-.022f;
    float cap=abs(p.y)-.62f;
    float da=max(max(radial,ribbonA),cap),db=max(max(radial+.014f,ribbonB),cap);
    which=step(db,da);edge=min(abs(ribbonA),abs(ribbonB));
    float bead=length(p-float3(.0f,.0f,.0f))-.115f;
    // Seven offset companion canes run along each broad ribbon.
    float wire=2.0f;
    for(int j=0;j<7;j++){
     float radius=.27f+float(j)*.03f,angle=p.y*5.4f;
     float2 v=radius*float2(cos(angle),sin(angle));
     wire=min(wire,max(length(p.xz-v)-.0045f,cap));
     wire=min(wire,max(length(p.xz+v)-.0045f,cap));
    }
    return min(min(min(da,db)*.52f,wire*.35f),bead);
}
static float hrmDE(float3 p,float t){float a,b;return hrmField(p,t,a,b);}
static float3 hrmNormal(float3 p,float t){const float e=.00045f;return normalize(float3(
 hrmDE(p+float3(e,0,0),t)-hrmDE(p-float3(e,0,0),t),
 hrmDE(p+float3(0,e,0),t)-hrmDE(p-float3(0,e,0),t),
 hrmDE(p+float3(0,0,e),t)-hrmDE(p-float3(0,0,e),t)));}

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
 if(!all(abs(ro)<DB_BOXDIMS-.001f)&&db_boxHit(ro,rd,DB_BOXDIMS,nn,true)<0)return float4(.006,.004,.012,1);
 ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize((u.patternTransform*float4(rd,0)).xyz);
 const float R=.84f;float b=dot(ro,rd),h=b*b-dot(ro,ro)+R*R;if(h<0)return float4(.006,.004,.012,1);
 float root=sqrt(h),entry=max(0.0f,-b-root);float3 sp=ro+rd*entry,sn=normalize(sp);
 float face=max(dot(-rd,sn),0.0f),fres=.04f+.96f*pow(1.0f-face,5.0f);
 float3 rr=refract(rd,sn,1.0f/1.47f);if(dot(ro,ro)<.84f*.84f||dot(rr,rr)<1e-8f)rr=rd;rr=normalize(rr);float3 rp=sp+rr*.004f;
 float rb=dot(rp,rr),rh=max(0.0f,rb*rb-dot(rp,rp)+R*R),end=-rb+sqrt(rh),z=0.0f;
 float3 col=float3(.018f,.014f,.040f)+float3(.24f,.40f,.68f)*fres*.35f;
 for(int i=0;i<288&&z<end;i++){float3 p=rp+rr*z;float id,edge,d=hrmField(p,u.time,id,edge);
  if(d<.00022f){float3 n=hrmNormal(p,u.time),l=normalize(float3(-.43f,.78f,.46f));float dif=max(dot(n,l),0.0f);
   float3 cyan=float3(.02f,.70f,.90f),ruby=float3(.92f,.035f,.16f),base=mix(cyan,ruby,id);
   base=mix(base,float3(1.0f,.70f,.22f),pow(1.0f-saturate(edge*24.0f),7.0f)*.28f);
   float filigree=pow(.5f+.5f*cos(length(p.xz)*340.0f+p.y*8.0f),14.0f);
   base=mix(base,float3(.97f,.83f,.43f),filigree*.45f);
   float spec=pow(max(dot(n,normalize(l-rr)),0.0f),72.0f);col+=base*(.23f+.95f*dif)+spec*float3(1,.88,.72);break;}
  z+=clamp(d*.48f,.00012f,.022f);
 }
 float glint=pow(max(dot(reflect(rd,sn),normalize(float3(-.43f,.78f,.46f))),0.0f),128.0f);
 col+=fres*float3(.55f,.82f,1.0f)*.62f+glint*float3(1,.94,.83);

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
