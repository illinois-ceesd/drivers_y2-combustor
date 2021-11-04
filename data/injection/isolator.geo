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
injectorsize = inj_t/20.; // background mesh size in the injector region

//Top Wall
//Point(1) = {0.2273,0.0270645,0.0,basesize};
Point(1) = {0.21,0.0270645,0.0,basesize};
Point(2) = {0.2280392417062,0.0270645,0.0,basesize};
Point(3) = {0.2287784834123,0.0270645,0.0,basesize};
Point(4) = {0.2295177251185,0.0270645,0.0,basesize};
Point(5) = {0.2302569668246,0.0270645,0.0,basesize};
Point(6) = {0.2309962085308,0.0270645,0.0,basesize};
Point(7) = {0.231735450237,0.0270645,0.0,basesize};
Point(8) = {0.2324746919431,0.0270645,0.0,basesize};
Point(9) = {0.2332139336493,0.0270645,0.0,basesize};
Point(10) = {0.2339531753555,0.0270645,0.0,basesize};
Point(11) = {0.2346924170616,0.02679523462424,0.0,basesize};
Point(12) = {0.2354316587678,0.02628798808666,0.0,basesize};
Point(13) = {0.2361709004739,0.02578074154909,0.0,basesize};
Point(14) = {0.2369101421801,0.02527349501151,0.0,basesize};
Point(15) = {0.2376493838863,0.02476624847393,0.0,basesize};
Point(16) = {0.2383886255924,0.02425900193636,0.0,basesize};
Point(17) = {0.2391278672986,0.02375175539878,0.0,basesize};
Point(18) = {0.2398671090047,0.02324450886121,0.0,basesize};
Point(19) = {0.2406063507109,0.02273726232363,0.0,basesize};
Point(20) = {0.2413455924171,0.02223001578605,0.0,basesize};
Point(21) = {0.2420848341232,0.02172276924848,0.0,basesize};
Point(22) = {0.2428240758294,0.0212155227109,0.0,basesize};
Point(23) = {0.2435633175355,0.02070827617332,0.0,basesize};
Point(24) = {0.2443025592417,0.02020102963575,0.0,basesize};
Point(25) = {0.2450418009479,0.01969378309817,0.0,basesize};
Point(26) = {0.245781042654,0.0191865365606,0.0,basesize};
Point(27) = {0.2465202843602,0.01867929002302,0.0,basesize};
Point(28) = {0.2472595260664,0.01817204348544,0.0,basesize};
Point(29) = {0.2479987677725,0.01766479694787,0.0,basesize};
Point(30) = {0.2487380094787,0.01715755041029,0.0,basesize};
Point(31) = {0.2494772511848,0.01665030387271,0.0,basesize};
Point(32) = {0.250216492891,0.01614305733514,0.0,basesize};
Point(33) = {0.2509557345972,0.01563581079756,0.0,basesize};
Point(34) = {0.2516949763033,0.01512856425999,0.0,basesize};
Point(35) = {0.2524342180095,0.01462131772241,0.0,basesize};
Point(36) = {0.2531734597156,0.01411407118483,0.0,basesize};
Point(37) = {0.2539127014218,0.01360682464726,0.0,basesize};
Point(38) = {0.254651943128,0.01309957810968,0.0,basesize};
Point(39) = {0.2553911848341,0.01259233157211,0.0,basesize};
Point(40) = {0.2561304265403,0.01208508503453,0.0,basesize};
Point(41) = {0.2568696682464,0.01157783849695,0.0,basesize};
Point(42) = {0.2576089099526,0.01107059195938,0.0,basesize};
Point(43) = {0.2583481516588,0.0105633454218,0.0,basesize};
Point(44) = {0.2590873933649,0.01005609888422,0.0,basesize};
Point(45) = {0.2598266350711,0.009548852346649,0.0,basesize};
Point(46) = {0.2605658767773,0.009041605809072,0.0,basesize};
Point(47) = {0.2613051184834,0.008534359271496,0.0,basesize};
Point(48) = {0.2620443601896,0.00802711273392,0.0,basesize};
Point(49) = {0.2627836018957,0.007519866196344,0.0,basesize};
Point(50) = {0.2635228436019,0.007012619658768,0.0,basesize};
Point(51) = {0.2642620853081,0.006505373121192,0.0,basesize};
Point(52) = {0.2650013270142,0.005998126583615,0.0,basesize};
Point(53) = {0.2657405687204,0.005490880046039,0.0,basesize};
Point(54) = {0.2664798104265,0.004983633508463,0.0,basesize};
Point(55) = {0.2672190521327,0.004476386970887,0.0,basesize};
Point(56) = {0.2679582938389,0.003969140433311,0.0,basesize};
Point(57) = {0.268697535545,0.003461893895735,0.0,basesize};
Point(58) = {0.2694367772512,0.002954647358158,0.0,basesize};
Point(59) = {0.2701760189573,0.002447400820582,0.0,basesize};
Point(60) = {0.2709152606635,0.001940154283006,0.0,basesize};
Point(61) = {0.2716545023697,0.00143290774543,0.0,basesize};
Point(62) = {0.2723937440758,0.0009256612078538,0.0,basesize};
Point(63) = {0.273132985782,0.0004184146702776,0.0,basesize};
Point(64) = {0.2738722274882,-8.883186729857e-05,0.0,basesize};
Point(65) = {0.2746114691943,-0.0005960784048747,0.0,basesize};
Point(66) = {0.2753507109005,-0.001103324942451,0.0,basesize};
Point(67) = {0.2760899526066,-0.001610571480027,0.0,basesize};
Point(68) = {0.2768291943128,-0.0021178180176,0.0,basesize};
Point(69) = {0.277568436019,-0.002625063418531,0.0,basesize};
Point(70) = {0.2783076777251,-0.003128071371827,0.0,basesize};
Point(71) = {0.2790469194313,-0.00356543025825,0.0,basesize};
Point(72) = {0.2797861611374,-0.003924485596916,0.0,basesize};
Point(73) = {0.2805254028436,-0.004209800511799,0.0,basesize};
Point(74) = {0.2812646445498,-0.004425962626834,0.0,basesize};
Point(75) = {0.2820038862559,-0.004577559566121,0.0,basesize};
Point(76) = {0.2827431279621,-0.004669178953759,0.0,basesize};
Point(77) = {0.2834823696682,-0.004705408413847,0.0,basesize};
Point(78) = {0.2842216113744,-0.004697204954745,0.0,basesize};
Point(79) = {0.2849608530806,-0.00465704436755,0.0,basesize};
Point(80) = {0.2857000947867,-0.004586244418798,0.0,basesize};
Point(81) = {0.2864393364929,-0.004485025473862,0.0,basesize};
Point(82) = {0.2871785781991,-0.004353607898117,0.0,basesize};
Point(83) = {0.2879178199052,-0.004192212056935,0.0,basesize};
Point(84) = {0.2886570616114,-0.00400105831569,0.0,basesize};
Point(85) = {0.2893963033175,-0.003780367039754,0.0,basesize};
Point(86) = {0.2901355450237,-0.003530358594502,0.0,basesize};
Point(87) = {0.2908747867299,-0.003251253345306,0.0,basesize};
Point(88) = {0.291614028436,-0.002943271657539,0.0,basesize};
Point(89) = {0.2923532701422,-0.002613060084159,0.0,basesize};
Point(90) = {0.2930925118483,-0.00228623916318,0.0,basesize};
Point(91) = {0.2938317535545,-0.001965379671836,0.0,basesize};
Point(92) = {0.2945709952607,-0.001650408524638,0.0,basesize};
Point(93) = {0.2953102369668,-0.001341252636095,0.0,basesize};
Point(94) = {0.296049478673,-0.001037838920719,0.0,basesize};
Point(95) = {0.2967887203791,-0.0007400942930211,0.0,basesize};
Point(96) = {0.2975279620853,-0.0004479456675107,0.0,basesize};
Point(97) = {0.2982672037915,-0.0001613199586989,0.0,basesize};
Point(98) = {0.2990064454976,0.0001198559189035,0.0,basesize};
Point(99) = {0.2997456872038,0.0003956550507858,0.0,basesize};
Point(100) = {0.30048492891,0.0006661505224375,0.0,basesize};
Point(101) = {0.3012241706161,0.0009314154193479,0.0,basesize};
Point(102) = {0.3019634123223,0.001191522827006,0.0,basesize};
Point(103) = {0.3027026540284,0.001446545830902,0.0,basesize};
Point(104) = {0.3034418957346,0.001696557516524,0.0,basesize};
Point(105) = {0.3041811374408,0.001941630969363,0.0,basesize};
Point(106) = {0.3049203791469,0.002181839274906,0.0,basesize};
Point(107) = {0.3056596208531,0.002417255518644,0.0,basesize};
Point(108) = {0.3063988625592,0.002647952786067,0.0,basesize};
Point(109) = {0.3071381042654,0.002874004162662,0.0,basesize};
Point(110) = {0.3078773459716,0.003095482733921,0.0,basesize};
Point(111) = {0.3086165876777,0.003312461585331,0.0,basesize};
Point(112) = {0.3093558293839,0.003525013802383,0.0,basesize};
Point(113) = {0.31009507109,0.003733212470565,0.0,basesize};
Point(114) = {0.3108343127962,0.003937130675367,0.0,basesize};
Point(115) = {0.3115735545024,0.004136841502279,0.0,basesize};
Point(116) = {0.3123127962085,0.004332418036789,0.0,basesize};
Point(117) = {0.3130520379147,0.004523933364387,0.0,basesize};
Point(118) = {0.3137912796209,0.004711460570563,0.0,basesize};
Point(119) = {0.314530521327,0.004895072740805,0.0,basesize};
Point(120) = {0.3152697630332,0.005074842960603,0.0,basesize};
Point(121) = {0.3160090047393,0.005250844315447,0.0,basesize};
Point(122) = {0.3167482464455,0.005423149890825,0.0,basesize};
Point(123) = {0.3174874881517,0.005591832772228,0.0,basesize};
Point(124) = {0.3182267298578,0.005756966045143,0.0,basesize};
Point(125) = {0.318965971564,0.005918622795062,0.0,basesize};
Point(126) = {0.3197052132701,0.006076876107472,0.0,basesize};
Point(127) = {0.3204444549763,0.006231799067864,0.0,basesize};
Point(128) = {0.3211836966825,0.006383464761726,0.0,basesize};
Point(129) = {0.3219229383886,0.006531946274548,0.0,basesize};
Point(130) = {0.3226621800948,0.00667731669182,0.0,basesize};
Point(131) = {0.3234014218009,0.00681964909903,0.0,basesize};
Point(132) = {0.3241406635071,0.006959016581669,0.0,basesize};
Point(133) = {0.3248799052133,0.007095492225225,0.0,basesize};
Point(134) = {0.3256191469194,0.007229149115187,0.0,basesize};
Point(135) = {0.3263583886256,0.007360060337045,0.0,basesize};
Point(136) = {0.3270976303318,0.007488298976289,0.0,basesize};
Point(137) = {0.3278368720379,0.007613938118408,0.0,basesize};
Point(138) = {0.3285761137441,0.00773705084889,0.0,basesize};
Point(139) = {0.3293153554502,0.007857710253226,0.0,basesize};
Point(140) = {0.3300545971564,0.007975989416904,0.0,basesize};
Point(141) = {0.3307938388626,0.008091961425415,0.0,basesize};
Point(142) = {0.3315330805687,0.008205699364247,0.0,basesize};
Point(143) = {0.3322723222749,0.008317276318889,0.0,basesize};
Point(144) = {0.333011563981,0.008426765374832,0.0,basesize};
Point(145) = {0.3337508056872,0.008534231284918,0.0,basesize};
Point(146) = {0.3344900473934,0.008639591517526,0.0,basesize};
Point(147) = {0.3352292890995,0.008742817528548,0.0,basesize};
Point(148) = {0.3359685308057,0.008843926036179,0.0,basesize};
Point(149) = {0.3367077725118,0.008942933758618,0.0,basesize};
Point(150) = {0.337447014218,0.009039857414062,0.0,basesize};
Point(151) = {0.3381862559242,0.00913471372071,0.0,basesize};
Point(152) = {0.3389254976303,0.009227519396758,0.0,basesize};
Point(153) = {0.3396647393365,0.009318291160404,0.0,basesize};
Point(154) = {0.3404039810427,0.009407045729847,0.0,basesize};
Point(155) = {0.3411432227488,0.009493799823283,0.0,basesize};
Point(156) = {0.341882464455,0.00957857015891,0.0,basesize};
Point(157) = {0.3426217061611,0.009661373454925,0.0,basesize};
Point(158) = {0.3433609478673,0.009742226429528,0.0,basesize};
Point(159) = {0.3441001895735,0.009821145800914,0.0,basesize};
Point(160) = {0.3448394312796,0.009898148287282,0.0,basesize};
Point(161) = {0.3455786729858,0.00997325060683,0.0,basesize};
Point(162) = {0.3463179146919,0.01004646947775,0.0,basesize};
Point(163) = {0.3470571563981,0.01011782161825,0.0,basesize};
Point(164) = {0.3477963981043,0.01018732374652,0.0,basesize};
Point(165) = {0.3485356398104,0.01025499258077,0.0,basesize};
Point(166) = {0.3492748815166,0.01032084483917,0.0,basesize};
Point(167) = {0.3500141232227,0.01038489723995,0.0,basesize};
Point(168) = {0.3507533649289,0.01044716650128,0.0,basesize};
Point(169) = {0.3514926066351,0.01050766934138,0.0,basesize};
Point(170) = {0.3522318483412,0.01056642247844,0.0,basesize};
Point(171) = {0.3529710900474,0.01062344263065,0.0,basesize};
Point(172) = {0.3537103317536,0.01067874651621,0.0,basesize};
Point(173) = {0.3544495734597,0.01073235085333,0.0,basesize};
Point(174) = {0.3551888151659,0.01078427236019,0.0,basesize};
Point(175) = {0.355928056872,0.010834527755,0.0,basesize};
Point(176) = {0.3566672985782,0.01088313375595,0.0,basesize};
Point(177) = {0.3574065402844,0.01093010708125,0.0,basesize};
Point(178) = {0.3581457819905,0.01097546444908,0.0,basesize};
Point(179) = {0.3588850236967,0.01101922257765,0.0,basesize};
Point(180) = {0.3596242654028,0.01106139818516,0.0,basesize};
Point(181) = {0.360363507109,0.0111020079898,0.0,basesize};
Point(182) = {0.3611027488152,0.01114106870976,0.0,basesize};
Point(183) = {0.3618419905213,0.01117859706326,0.0,basesize};
Point(184) = {0.3625812322275,0.01121460976848,0.0,basesize};
Point(185) = {0.3633204739336,0.01124912354362,0.0,basesize};
Point(186) = {0.3640597156398,0.01128215510688,0.0,basesize};
Point(187) = {0.364798957346,0.01131372117646,0.0,basesize};
Point(188) = {0.3655381990521,0.01134383847056,0.0,basesize};
Point(189) = {0.3662774407583,0.01137252370737,0.0,basesize};
Point(190) = {0.3670166824645,0.01139979360508,0.0,basesize};
Point(191) = {0.3677559241706,0.01142566488191,0.0,basesize};
Point(192) = {0.3684951658768,0.01145015425605,0.0,basesize};
Point(193) = {0.3692344075829,0.01147327844568,0.0,basesize};
Point(194) = {0.3699736492891,0.01149505416902,0.0,basesize};
Point(195) = {0.3707128909953,0.01151549814426,0.0,basesize};
Point(196) = {0.3714521327014,0.01153462708959,0.0,basesize};
Point(197) = {0.3721913744076,0.01155245772322,0.0,basesize};
Point(198) = {0.3729306161137,0.01156900676334,0.0,basesize};
Point(199) = {0.3736698578199,0.01158429092815,0.0,basesize};
Point(200) = {0.3744090995261,0.01159832693585,0.0,basesize};
Point(201) = {0.3751483412322,0.01161113150463,0.0,basesize};
Point(202) = {0.3758875829384,0.01162272135269,0.0,basesize};
Point(203) = {0.3766268246445,0.01163311319823,0.0,basesize};
Point(204) = {0.3773660663507,0.01164232375945,0.0,basesize};
Point(205) = {0.3781053080569,0.01165036975455,0.0,basesize};
Point(206) = {0.378844549763,0.01165726790172,0.0,basesize};
Point(207) = {0.3795837914692,0.01166303491916,0.0,basesize};
Point(208) = {0.3803230331754,0.01166768752507,0.0,basesize};
Point(209) = {0.3810622748815,0.01167124243764,0.0,basesize};
Point(210) = {0.3818015165877,0.01167371637508,0.0,basesize};
Point(211) = {0.3825407582938,0.01167512605558,0.0,basesize};
Point(212) = {0.38328,0.01167548819733,0.0,basesize};

//Bottom Wall
//Point(213) = {0.2273,-0.0270645,0.0,basesize};
Point(213) = {0.21,-0.0270645,0.0,basesize};
Point(214) = {0.2280392417062,-0.0270645,0.0,basesize};
Point(215) = {0.2287784834123,-0.0270645,0.0,basesize};
Point(216) = {0.2295177251185,-0.0270645,0.0,basesize};
Point(217) = {0.2302569668246,-0.0270645,0.0,basesize};
Point(218) = {0.2309962085308,-0.0270645,0.0,basesize};
Point(219) = {0.231735450237,-0.0270645,0.0,basesize};
Point(220) = {0.2324746919431,-0.0270645,0.0,basesize};
Point(221) = {0.2332139336493,-0.0270645,0.0,basesize};
Point(222) = {0.2339531753555,-0.0270645,0.0,basesize};
Point(223) = {0.2346924170616,-0.02679430246686,0.0,basesize};
Point(224) = {0.2354316587678,-0.0262852999159,0.0,basesize};
Point(225) = {0.2361709004739,-0.02577629736494,0.0,basesize};
Point(226) = {0.2369101421801,-0.02526729481398,0.0,basesize};
Point(227) = {0.2376493838863,-0.02475829226302,0.0,basesize};
Point(228) = {0.2383886255924,-0.02424928971206,0.0,basesize};
Point(229) = {0.2391278672986,-0.0237402871611,0.0,basesize};
Point(230) = {0.2398671090047,-0.02323128461014,0.0,basesize};
Point(231) = {0.2406063507109,-0.02272228205918,0.0,basesize};
Point(232) = {0.2413455924171,-0.02221327950822,0.0,basesize};
Point(233) = {0.2420848341232,-0.02170427695726,0.0,basesize};
Point(234) = {0.2428240758294,-0.0211952744063,0.0,basesize};
Point(235) = {0.2435633175355,-0.02068627185534,0.0,basesize};
Point(236) = {0.2443025592417,-0.02017726930438,0.0,basesize};
Point(237) = {0.2450418009479,-0.01966826675342,0.0,basesize};
Point(238) = {0.245781042654,-0.01915926420245,0.0,basesize};
Point(239) = {0.2465202843602,-0.01865026165149,0.0,basesize};
Point(240) = {0.2472595260664,-0.01814125910053,0.0,basesize};
Point(241) = {0.2479987677725,-0.01763225654957,0.0,basesize};
Point(242) = {0.2487380094787,-0.01712325399861,0.0,basesize};
Point(243) = {0.2494772511848,-0.01661425144765,0.0,basesize};
Point(244) = {0.250216492891,-0.01610524889669,0.0,basesize};
Point(245) = {0.2509557345972,-0.01559624634573,0.0,basesize};
Point(246) = {0.2516949763033,-0.01508724379477,0.0,basesize};
Point(247) = {0.2524342180095,-0.01457824124381,0.0,basesize};
Point(248) = {0.2531734597156,-0.01406923869285,0.0,basesize};
Point(249) = {0.2539127014218,-0.01356023614189,0.0,basesize};
Point(250) = {0.254651943128,-0.01305123359093,0.0,basesize};
Point(251) = {0.2553911848341,-0.01254223104006,0.0,basesize};
Point(252) = {0.2561304265403,-0.01203324300793,0.0,basesize};
Point(253) = {0.2568696682464,-0.01153323105766,0.0,basesize};
Point(254) = {0.2576089099526,-0.01107308263704,0.0,basesize};
Point(255) = {0.2583481516588,-0.01065403753526,0.0,basesize};
Point(256) = {0.2590873933649,-0.0102747284778,0.0,basesize};
Point(257) = {0.2598266350711,-0.009933787576145,0.0,basesize};
Point(258) = {0.2605658767773,-0.009629846941815,0.0,basesize};
Point(259) = {0.2613051184834,-0.009361538686311,0.0,basesize};
Point(260) = {0.2620443601896,-0.009127494921139,0.0,basesize};
Point(261) = {0.2627836018957,-0.008926347757803,0.0,basesize};
Point(262) = {0.2635228436019,-0.008756729307808,0.0,basesize};
Point(263) = {0.2642620853081,-0.008617271682661,0.0,basesize};
Point(264) = {0.2650013270142,-0.008506606993866,0.0,basesize};
Point(265) = {0.2657405687204,-0.008423367352927,0.0,basesize};
Point(266) = {0.2664798104265,-0.008366184871351,0.0,basesize};
Point(267) = {0.2672190521327,-0.008333691646977,0.0,basesize};
Point(268) = {0.2679582938389,-0.008324504031647,0.0,basesize};
Point(269) = {0.268697535545,-0.008324500001239,0.0,basesize};
Point(270) = {0.2694367772512,-0.0083245,0.0,basesize};
Point(271) = {0.2701760189573,-0.0083245,0.0,basesize};
Point(272) = {0.2709152606635,-0.0083245,0.0,basesize};
Point(273) = {0.2716545023697,-0.0083245,0.0,basesize};
Point(274) = {0.2723937440758,-0.0083245,0.0,basesize};
Point(275) = {0.273132985782,-0.0083245,0.0,basesize};
Point(276) = {0.2738722274882,-0.0083245,0.0,basesize};
Point(277) = {0.2746114691943,-0.0083245,0.0,basesize};
Point(278) = {0.2753507109005,-0.0083245,0.0,basesize};
Point(279) = {0.2760899526066,-0.0083245,0.0,basesize};
Point(280) = {0.2768291943128,-0.0083245,0.0,basesize};
Point(281) = {0.277568436019,-0.0083245,0.0,basesize};
Point(282) = {0.2783076777251,-0.0083245,0.0,basesize};
Point(283) = {0.2790469194313,-0.0083245,0.0,basesize};
Point(284) = {0.2797861611374,-0.0083245,0.0,basesize};
Point(285) = {0.2805254028436,-0.0083245,0.0,basesize};
Point(286) = {0.2812646445498,-0.0083245,0.0,basesize};
Point(287) = {0.2820038862559,-0.0083245,0.0,basesize};
Point(288) = {0.2827431279621,-0.0083245,0.0,basesize};
Point(289) = {0.2834823696682,-0.0083245,0.0,basesize};
Point(290) = {0.2842216113744,-0.0083245,0.0,basesize};
Point(291) = {0.2849608530806,-0.0083245,0.0,basesize};
Point(292) = {0.2857000947867,-0.0083245,0.0,basesize};
Point(293) = {0.2864393364929,-0.0083245,0.0,basesize};
Point(294) = {0.2871785781991,-0.0083245,0.0,basesize};
Point(295) = {0.2879178199052,-0.0083245,0.0,basesize};
Point(296) = {0.2886570616114,-0.0083245,0.0,basesize};
Point(297) = {0.2893963033175,-0.0083245,0.0,basesize};
Point(298) = {0.2901355450237,-0.0083245,0.0,basesize};
Point(299) = {0.2908747867299,-0.0083245,0.0,basesize};
Point(300) = {0.291614028436,-0.0083245,0.0,basesize};
Point(301) = {0.2923532701422,-0.0083245,0.0,basesize};
Point(302) = {0.2930925118483,-0.0083245,0.0,basesize};
Point(303) = {0.2938317535545,-0.0083245,0.0,basesize};
Point(304) = {0.2945709952607,-0.0083245,0.0,basesize};
Point(305) = {0.2953102369668,-0.0083245,0.0,basesize};
Point(306) = {0.296049478673,-0.0083245,0.0,basesize};
Point(307) = {0.2967887203791,-0.0083245,0.0,basesize};
Point(308) = {0.2975279620853,-0.0083245,0.0,basesize};
Point(309) = {0.2982672037915,-0.0083245,0.0,basesize};
Point(310) = {0.2990064454976,-0.0083245,0.0,basesize};
Point(311) = {0.2997456872038,-0.0083245,0.0,basesize};
Point(312) = {0.30048492891,-0.0083245,0.0,basesize};
Point(313) = {0.3012241706161,-0.0083245,0.0,basesize};
Point(314) = {0.3019634123223,-0.0083245,0.0,basesize};
Point(315) = {0.3027026540284,-0.0083245,0.0,basesize};
Point(316) = {0.3034418957346,-0.0083245,0.0,basesize};
Point(317) = {0.3041811374408,-0.0083245,0.0,basesize};
Point(318) = {0.3049203791469,-0.0083245,0.0,basesize};
Point(319) = {0.3056596208531,-0.0083245,0.0,basesize};
Point(320) = {0.3063988625592,-0.0083245,0.0,basesize};
Point(321) = {0.3071381042654,-0.0083245,0.0,basesize};
Point(322) = {0.3078773459716,-0.0083245,0.0,basesize};
Point(323) = {0.3086165876777,-0.0083245,0.0,basesize};
Point(324) = {0.3093558293839,-0.0083245,0.0,basesize};
Point(325) = {0.31009507109,-0.0083245,0.0,basesize};
Point(326) = {0.3108343127962,-0.0083245,0.0,basesize};
Point(327) = {0.3115735545024,-0.0083245,0.0,basesize};
Point(328) = {0.3123127962085,-0.0083245,0.0,basesize};
Point(329) = {0.3130520379147,-0.0083245,0.0,basesize};
Point(330) = {0.3137912796209,-0.0083245,0.0,basesize};
Point(331) = {0.314530521327,-0.0083245,0.0,basesize};
Point(332) = {0.3152697630332,-0.0083245,0.0,basesize};
Point(333) = {0.3160090047393,-0.0083245,0.0,basesize};
Point(334) = {0.3167482464455,-0.0083245,0.0,basesize};
Point(335) = {0.3174874881517,-0.0083245,0.0,basesize};
Point(336) = {0.3182267298578,-0.0083245,0.0,basesize};
Point(337) = {0.318965971564,-0.0083245,0.0,basesize};
Point(338) = {0.3197052132701,-0.0083245,0.0,basesize};
Point(339) = {0.3204444549763,-0.0083245,0.0,basesize};
Point(340) = {0.3211836966825,-0.0083245,0.0,basesize};
Point(341) = {0.3219229383886,-0.0083245,0.0,basesize};
Point(342) = {0.3226621800948,-0.0083245,0.0,basesize};
Point(343) = {0.3234014218009,-0.0083245,0.0,basesize};
Point(344) = {0.3241406635071,-0.0083245,0.0,basesize};
Point(345) = {0.3248799052133,-0.0083245,0.0,basesize};
Point(346) = {0.3256191469194,-0.0083245,0.0,basesize};
Point(347) = {0.3263583886256,-0.0083245,0.0,basesize};
Point(348) = {0.3270976303318,-0.0083245,0.0,basesize};
Point(349) = {0.3278368720379,-0.0083245,0.0,basesize};
Point(350) = {0.3285761137441,-0.0083245,0.0,basesize};
Point(351) = {0.3293153554502,-0.0083245,0.0,basesize};
Point(352) = {0.3300545971564,-0.0083245,0.0,basesize};
Point(353) = {0.3307938388626,-0.0083245,0.0,basesize};
Point(354) = {0.3315330805687,-0.0083245,0.0,basesize};
Point(355) = {0.3322723222749,-0.0083245,0.0,basesize};
Point(356) = {0.333011563981,-0.0083245,0.0,basesize};
Point(357) = {0.3337508056872,-0.0083245,0.0,basesize};
Point(358) = {0.3344900473934,-0.0083245,0.0,basesize};
Point(359) = {0.3352292890995,-0.0083245,0.0,basesize};
Point(360) = {0.3359685308057,-0.0083245,0.0,basesize};
Point(361) = {0.3367077725118,-0.0083245,0.0,basesize};
Point(362) = {0.337447014218,-0.0083245,0.0,basesize};
Point(363) = {0.3381862559242,-0.0083245,0.0,basesize};
Point(364) = {0.3389254976303,-0.0083245,0.0,basesize};
Point(365) = {0.3396647393365,-0.0083245,0.0,basesize};
Point(366) = {0.3404039810427,-0.0083245,0.0,basesize};
Point(367) = {0.3411432227488,-0.0083245,0.0,basesize};
Point(368) = {0.341882464455,-0.0083245,0.0,basesize};
Point(369) = {0.3426217061611,-0.0083245,0.0,basesize};
Point(370) = {0.3433609478673,-0.0083245,0.0,basesize};
Point(371) = {0.3441001895735,-0.0083245,0.0,basesize};
Point(372) = {0.3448394312796,-0.0083245,0.0,basesize};
Point(373) = {0.3455786729858,-0.0083245,0.0,basesize};
Point(374) = {0.3463179146919,-0.0083245,0.0,basesize};
Point(375) = {0.3470571563981,-0.0083245,0.0,basesize};
Point(376) = {0.3477963981043,-0.0083245,0.0,basesize};
Point(377) = {0.3485356398104,-0.0083245,0.0,basesize};
Point(378) = {0.3492748815166,-0.0083245,0.0,basesize};
Point(379) = {0.3500141232227,-0.0083245,0.0,basesize};
Point(380) = {0.3507533649289,-0.0083245,0.0,basesize};
Point(381) = {0.3514926066351,-0.0083245,0.0,basesize};
Point(382) = {0.3522318483412,-0.0083245,0.0,basesize};
Point(383) = {0.3529710900474,-0.0083245,0.0,basesize};
Point(384) = {0.3537103317536,-0.0083245,0.0,basesize};
Point(385) = {0.3544495734597,-0.0083245,0.0,basesize};
Point(386) = {0.3551888151659,-0.0083245,0.0,basesize};
Point(387) = {0.355928056872,-0.0083245,0.0,basesize};
Point(388) = {0.3566672985782,-0.0083245,0.0,basesize};
Point(389) = {0.3574065402844,-0.0083245,0.0,basesize};
Point(390) = {0.3581457819905,-0.0083245,0.0,basesize};
Point(391) = {0.3588850236967,-0.0083245,0.0,basesize};
Point(392) = {0.3596242654028,-0.0083245,0.0,basesize};
Point(393) = {0.360363507109,-0.0083245,0.0,basesize};
Point(394) = {0.3611027488152,-0.0083245,0.0,basesize};
Point(395) = {0.3618419905213,-0.0083245,0.0,basesize};
Point(396) = {0.3625812322275,-0.0083245,0.0,basesize};
Point(397) = {0.3633204739336,-0.0083245,0.0,basesize};
Point(398) = {0.3640597156398,-0.0083245,0.0,basesize};
Point(399) = {0.364798957346,-0.0083245,0.0,basesize};
Point(400) = {0.3655381990521,-0.0083245,0.0,basesize};
Point(401) = {0.3662774407583,-0.0083245,0.0,basesize};
Point(402) = {0.3670166824645,-0.0083245,0.0,basesize};
Point(403) = {0.3677559241706,-0.0083245,0.0,basesize};
Point(404) = {0.3684951658768,-0.0083245,0.0,basesize};
Point(405) = {0.3692344075829,-0.0083245,0.0,basesize};
Point(406) = {0.3699736492891,-0.0083245,0.0,basesize};
Point(407) = {0.3707128909953,-0.0083245,0.0,basesize};
Point(408) = {0.3714521327014,-0.0083245,0.0,basesize};
Point(409) = {0.3721913744076,-0.0083245,0.0,basesize};
Point(410) = {0.3729306161137,-0.0083245,0.0,basesize};
Point(411) = {0.3736698578199,-0.0083245,0.0,basesize};
Point(412) = {0.3744090995261,-0.0083245,0.0,basesize};
Point(413) = {0.3751483412322,-0.0083245,0.0,basesize};
Point(414) = {0.3758875829384,-0.0083245,0.0,basesize};
Point(415) = {0.3766268246445,-0.0083245,0.0,basesize};
Point(416) = {0.3773660663507,-0.0083245,0.0,basesize};
Point(417) = {0.3781053080569,-0.0083245,0.0,basesize};
Point(418) = {0.378844549763,-0.0083245,0.0,basesize};
Point(419) = {0.3795837914692,-0.0083245,0.0,basesize};
Point(420) = {0.3803230331754,-0.0083245,0.0,basesize};
Point(421) = {0.3810622748815,-0.0083245,0.0,basesize};
Point(422) = {0.3818015165877,-0.0083245,0.0,basesize};
Point(423) = {0.38328,-0.0083245,0.0,basesize};

//Make Lines
//Top
BSpline(1000) = {1:212};  // goes clockwise

//Bottom Lines
BSpline(1001) = {213:423}; // goes counter-clockwise

//Mid-point on inlet
Point(460) = {0.21, (0.0270645-0.0270645)/2.,0.0,2*basesize};
//Inlet
Line(423) = {1,213};  //goes counter-clockwise
//Line(458) = {1,460};
//Line(459) = {460,213};

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
Point(455) = {0.65163+0.335,-0.008324-(0.265-0.02)*Sin(2*Pi/180),0.0,basesize};
Point(456) = {0.65163+0.335,0.01167548819733,0.0,basesize};

//Point(500) = {0.70163+inj_h*Tan(cavityAngle*Pi/180), -0.0283245+inj_h, 0., basesize};
//Point(501) = {0.70163+inj_h*Tan(cavityAngle*Pi/180)+inj_d, -0.0283245+inj_h, 0., basesize};
//Point(502) = {0.70163+inj_h*Tan(cavityAngle*Pi/180)+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
//Point(503) = {0.70163+(inj_h+inj_t)*Tan(cavityAngle*Pi/180), -0.0283245+inj_h+inj_t, 0., basesize};

Point(500) = {0.70163+inj_h, -0.0283245+inj_h, 0., basesize};
Point(501) = {0.70163+inj_h+inj_d, -0.0283245+inj_h, 0., basesize};
Point(502) = {0.70163+inj_h+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
Point(503) = {0.70163+inj_h+inj_t, -0.0283245+inj_h+inj_t, 0., basesize};


//Make Cavity lines
Line(450) = {423,450};
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
Line(457) = {456,212};  // goes counter-clockwise

//Create lineloop of this geometry
// start on the bottom left and go around clockwise
Curve Loop(1) = {
-423, // inlet
1000, // top wall
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
-450, // isolator to cavity
-1001 // bottom wall
};

Plane Surface(1) = {1};

Physical Surface('domain') = {1};

Physical Curve('inflow') = {-423};
Physical Curve('injection') = {-502};
Physical Curve('outflow') = {456};
Physical Curve('wall') = {
//1:211,
1000,
//213:422,
1001,
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
Field[1].CurvesList = {1000,1001,450,454,455,457};
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
injector_bottom = -0.0235;
injector_top = -0.0225;
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
