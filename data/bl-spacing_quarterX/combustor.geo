If(Exists(size))
    basesize=size;
Else
    basesize=0.0002;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=1.0;
EndIf

If(Exists(inj_ratio))
    injector_ratio=inj_ratio;
Else
    injector_ratio=1.0;
EndIf

// horizontal injection
cavityAngle=45;
inj_h=4.e-3;  // height of injector (bottom) from floor
inj_t=1.59e-3; // diameter of injector
inj_d = 20e-3; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
isosize = basesize;       // background mesh size in the isolator
nozzlesize = basesize/2;       // background mesh size in the isolator
cavitysize = basesize/2.; // background mesh size in the cavity region
injectorsize = inj_t/10./injector_ratio; // background mesh size in the injector region

Point(420) = {0.6,0.01167548819733,0.0,basesize};
Point(430) = {0.6,-0.0083245,0.0,basesize};

//Make Lines
// Inlet
Line(423) = {420,430};  //goes counter-clockwise

//Cavity Start
Point(450) = {0.65163,-0.0083245,0.0,basesize};


//Bottom of cavity
Point(451) = {0.65163,-0.0283245,0.0,basesize};
Point(452) = {0.70163,-0.0283245,0.0,basesize};
Point(453) = {0.72163,-0.0083245,0.0,basesize};
Point(454) = {0.72163+0.02,-0.0083245,0.0,basesize};

//Extend downstream a bit
//Point(454) = {0.872163,-0.0083245,0.0,basesize};
//Point(455) = {0.872163,0.01167548819733,0.0,basesize};
Point(455) = {0.65163+0.12,-0.008324,0.0,basesize};
Point(456) = {0.65163+0.12,0.01167548819733,0.0,basesize};

//Point(500) = {0.70163+inj_h*Tan(cavityAngle*Pi/180), -0.0283245+inj_h, 0., basesize};
//Point(501) = {0.70163+inj_h*Tan(cavityAngle*Pi/180)+inj_d, -0.0283245+inj_h, 0., basesize};
//Point(502) = {0.70163+inj_h*Tan(cavityAngle*Pi/180)+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
//Point(503) = {0.70163+(inj_h+inj_t)*Tan(cavityAngle*Pi/180), -0.0283245+inj_h+inj_t, 0., basesize};

Point(500) = {0.70163+inj_h, -0.0283245+inj_h, 0., basesize};
Point(501) = {0.70163+inj_h+inj_d, -0.0283245+inj_h, 0., basesize};
Point(502) = {0.70163+inj_h+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
Point(503) = {0.70163+inj_h+inj_t, -0.0283245+inj_h+inj_t, 0., basesize};


//Make Cavity lines
Line(450) = {430,450};
Line(451) = {450,451};
Line(452) = {451,452};
Line(500) = {452,500};
Line(453) = {503,453};
Line(454) = {453,454};
Line(455) = {454,455};
// injector
Line(501) = {500,501};
Line(502) = {501,502};  // injector inlet
Line(503) = {502,503};
//Outlet
Line(456) = {455,456};
//Top wall
//Line(457) = {212,456};  // goes clockwise
Line(457) = {456,420};  // goes counter-clockwise

//Create lineloop of this geometry
// start on the bottom left and go around clockwise
Curve Loop(1) = {
-423, // inlet
-457, // extension to end
-456, // outlet
-455, // bottom expansion
-454, // post-cavity flat
-453, // cavity rear upper (slant)
-503, // injector top
-502, // injector inlet (slant)
-501, // injector bottom (slant)
-500, // cavity rear lower (slant)
-452, // cavity bottom
-451, // cavity front
-450 // isolator to cavity
};

Plane Surface(1) = {1};

Physical Surface('domain') = {1};

Physical Curve('inflow') = {-423};
Physical Curve('injection') = {-502};
Physical Curve('outflow') = {456};
Physical Curve('wall') = {
450,
451,
452,
453,
454,
455,
457,
500,
501,
503
};

// Create distance field from curves, excludes cavity
Field[1] = Distance;
Field[1].CurvesList = {450,454,455,457};
Field[1].NumPointsPerCurve = 100000;

// transfer the distance into something that goes from 
//Field[50] = MathEval;
//Field[50].F = Sprintf("F4^3 + %g", lc / 100);

//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = nozzlesize / boundratio;
Field[2].SizeMax = isosize;
Field[2].DistMin = 0.0002;
Field[2].DistMax = 0.005;
Field[2].StopAtDistMax = 1;

// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].CurvesList = {451:453};
Field[11].NumPointsPerCurve = 100000;

//Create threshold field that varrries element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratio;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.0002;
Field[12].DistMax = 0.005;
Field[12].StopAtDistMax = 1;

nozzle_start = 0.27;
nozzle_end = 0.30;
//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = nozzle_end;
Field[3].XMax = 1.0;
Field[3].YMin = -1.0;
Field[3].YMax = 1.0;
Field[3].VIn = isosize;
Field[3].VOut = bigsize;

// background mesh size upstream of the inlet
Field[4] = Box;
Field[4].XMin = 0.;
Field[4].XMax = nozzle_start;
Field[4].YMin = -1.0;
Field[4].YMax = 1.0;
Field[4].VIn = inletsize;
Field[4].VOut = bigsize;

// background mesh size in the nozzle throat
Field[5] = Box;
Field[5].XMin = nozzle_start;
Field[5].XMax = nozzle_end;
Field[5].YMin = -1.0;
Field[5].YMax = 1.0;
Field[5].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[5].VIn = nozzlesize;
Field[5].VOut = bigsize;

// background mesh size in the cavity region
cavity_start = 0.65;
cavity_end = 0.73;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1.0;
Field[6].YMax = -0.003;
Field[6].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;

// background mesh size for the injector
injector_start = 0.69;
injector_end = 0.75;
injector_bottom = -0.022;
injector_top = -0.025;
Field[7] = Box;
Field[7].XMin = injector_start;
Field[7].XMax = injector_end;
Field[7].YMin = injector_bottom;
Field[7].YMax = injector_top;
Field[7].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;


// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {2, 3, 4, 5, 6, 7};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

//Mesh.Smoothing = 3;
