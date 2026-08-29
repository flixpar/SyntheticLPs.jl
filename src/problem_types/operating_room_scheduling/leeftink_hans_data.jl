# Compact empirical duration profiles derived from the public Leeftink--Hans
# operating-room scheduling benchmark (2019 release).  The source files contain
# more than 1,000 fitted three-parameter lognormal surgery types.  To keep this
# package small and self-contained, each specialty below retains the weighted
# 1/12, 3/12, ..., 11/12 quantile representatives of expected duration.  A
# repeated representative is intentional: it records a high-frequency surgery
# type rather than inventing a nearby synthetic type.
#
# Parameters use the benchmark convention
#     duration = gamma + LogNormal(mu, sigma),
# and `aggregate_mean`/`aggregate_cv` are moments of the complete specialty
# file, not just the six representatives.  Relative specialty weights and the
# clinical day-case/LOS fields are documented synthetic assumptions because
# the benchmark normalizes every specialty file separately and contains no
# urgency or downstream-bed data.
const _ORSCHED_SPECIALTIES = (
    (code=:CHI, name=:general_surgery, aggregate_mean=145.7, aggregate_cv=0.736,
     weight=1.2, day_case=0.45, ward_los=(1, 4), icu=0.08,
     types=(
        (id=1155, mu=4.012172857, sigma=0.225746247, gamma=0.0),
        (id=1021, mu=4.350849218, sigma=0.730968513, gamma=9.995941464),
        (id=1021, mu=4.350849218, sigma=0.730968513, gamma=9.995941464),
        (id=1157, mu=4.982424130, sigma=0.522129511, gamma=9.998655271),
        (id=1157, mu=4.982424130, sigma=0.522129511, gamma=9.998655271),
        (id=1080, mu=5.413303874, sigma=0.224370526, gamma=5.115645223),
     )),
    (code=:ENT, name=:otolaryngology, aggregate_mean=127.7, aggregate_cv=0.896,
     weight=1.0, day_case=0.75, ward_los=(0, 2), icu=0.02,
     types=(
        (id=1257, mu=3.770117322, sigma=0.332437531, gamma=2.654521776),
        (id=1158, mu=3.193383228, sigma=1.062027184, gamma=29.999900000),
        (id=1253, mu=4.572972744, sigma=0.477803187, gamma=0.0),
        (id=1281, mu=4.484873283, sigma=0.769315436, gamma=9.996371142),
        (id=1168, mu=5.023837511, sigma=0.565451677, gamma=9.998621980),
        (id=1252, mu=4.907150047, sigma=0.851025410, gamma=9.997355164),
     )),
    (code=:EYE, name=:ophthalmology, aggregate_mean=66.0, aggregate_cv=0.581,
     weight=1.3, day_case=0.95, ward_los=(0, 1), icu=0.00,
     types=(
        (id=1331, mu=3.485296772, sigma=0.374769910, gamma=5.667281020),
        (id=1284, mu=3.249110331, sigma=0.571881296, gamma=14.999900000),
        (id=1307, mu=2.986693620, sigma=0.890549648, gamma=28.588398430),
        (id=1351, mu=3.571146117, sigma=0.646390639, gamma=24.999900000),
        (id=1357, mu=4.342605759, sigma=0.452856797, gamma=0.0),
        (id=1324, mu=3.832255719, sigma=0.542846439, gamma=42.983205710),
     )),
    (code=:GYN, name=:gynecology, aggregate_mean=94.3, aggregate_cv=0.778,
     weight=0.9, day_case=0.70, ward_los=(0, 3), icu=0.03,
     types=(
        (id=1410, mu=3.844094746, sigma=0.328493275, gamma=5.376779167),
        (id=1425, mu=4.136559865, sigma=0.690021972, gamma=9.995090886),
        (id=1425, mu=4.136559865, sigma=0.690021972, gamma=9.995090886),
        (id=1425, mu=4.136559865, sigma=0.690021972, gamma=9.995090886),
        (id=1425, mu=4.136559865, sigma=0.690021972, gamma=9.995090886),
        (id=1381, mu=4.151856705, sigma=0.626122961, gamma=45.999900000),
     )),
    (code=:MIX, name=:mixed_specialties, aggregate_mean=78.9, aggregate_cv=1.093,
     weight=0.8, day_case=0.85, ward_los=(0, 2), icu=0.02,
     types=(
        (id=1427, mu=2.614994291, sigma=0.392378062, gamma=6.997833995),
        (id=1453, mu=2.675412982, sigma=0.533927890, gamma=16.985879820),
        (id=1485, mu=3.010285254, sigma=0.664430340, gamma=24.999900000),
        (id=1526, mu=3.590678853, sigma=0.462908858, gamma=35.999900000),
        (id=1652, mu=4.147838762, sigma=0.739916221, gamma=2.999900000),
        (id=1580, mu=4.916827354, sigma=0.838749053, gamma=9.999900000),
     )),
    (code=:NEU, name=:neurosurgery, aggregate_mean=94.8, aggregate_cv=1.298,
     weight=0.5, day_case=0.05, ward_los=(3, 7), icu=0.50,
     types=(
        (id=1688, mu=1.850690611, sigma=0.449038546, gamma=5.952754756),
        (id=1658, mu=2.117072622, sigma=0.858148205, gamma=5.999900000),
        (id=1698, mu=2.576054944, sigma=0.738938277, gamma=2.948019118),
        (id=1699, mu=3.069154746, sigma=0.597782155, gamma=10.999900000),
        (id=1661, mu=5.223848622, sigma=0.537587853, gamma=9.998977951),
        (id=1661, mu=5.223848622, sigma=0.537587853, gamma=9.998977951),
     )),
    (code=:ONC, name=:oncology, aggregate_mean=138.8, aggregate_cv=0.848,
     weight=0.6, day_case=0.20, ward_los=(2, 6), icu=0.20,
     types=(
        (id=1711, mu=4.019107332, sigma=0.483372188, gamma=0.0),
        (id=1729, mu=4.190917662, sigma=0.550275915, gamma=9.996178368),
        (id=1726, mu=4.535940277, sigma=0.610757267, gamma=9.997309849),
        (id=1703, mu=4.714475069, sigma=0.702704718, gamma=9.997421086),
        (id=1716, mu=4.838409332, sigma=0.601203465, gamma=9.998128201),
        (id=1704, mu=5.165666410, sigma=0.773011359, gamma=9.998358056),
     )),
    (code=:ORT, name=:orthopedics, aggregate_mean=87.5, aggregate_cv=0.720,
     weight=1.1, day_case=0.25, ward_los=(2, 5), icu=0.08,
     types=(
        (id=1800, mu=2.600263906, sigma=0.568653909, gamma=23.997665720),
        (id=1863, mu=3.789568669, sigma=0.379285205, gamma=0.0),
        (id=1807, mu=3.468320409, sigma=0.609141830, gamma=28.999900000),
        (id=1820, mu=3.792479434, sigma=0.312595743, gamma=43.999900000),
        (id=1809, mu=3.422189212, sigma=0.479493947, gamma=64.971341490),
        (id=1766, mu=4.021792902, sigma=0.456114391, gamma=102.997513300),
     )),
    (code=:PLA, name=:plastic_surgery, aggregate_mean=86.8, aggregate_cv=0.884,
     weight=0.8, day_case=0.55, ward_los=(1, 4), icu=0.03,
     types=(
        (id=1909, mu=3.484291429, sigma=0.463327082, gamma=2.729712661),
        (id=1909, mu=3.484291429, sigma=0.463327082, gamma=2.729712661),
        (id=1909, mu=3.484291429, sigma=0.463327082, gamma=2.729712661),
        (id=1871, mu=3.706390911, sigma=0.544191968, gamma=35.999900000),
        (id=1910, mu=4.600809381, sigma=0.522891594, gamma=0.0),
        (id=1897, mu=3.830541588, sigma=1.041905597, gamma=96.999900000),
     )),
    (code=:THO, name=:thoracic_surgery, aggregate_mean=174.2, aggregate_cv=0.768,
     weight=0.4, day_case=0.00, ward_los=(4, 8), icu=0.75,
     types=(
        (id=1959, mu=2.134863879, sigma=0.872927959, gamma=5.999900000),
        (id=1949, mu=3.988563733, sigma=0.472460278, gamma=0.0),
        (id=1938, mu=4.975343317, sigma=0.464372555, gamma=9.998859814),
        (id=1957, mu=5.457137106, sigma=0.468426553, gamma=0.0),
        (id=1935, mu=5.505607987, sigma=0.387499829, gamma=0.0),
        (id=1935, mu=5.505607987, sigma=0.387499829, gamma=0.0),
     )),
    (code=:URO, name=:urology, aggregate_mean=78.5, aggregate_cv=0.865,
     weight=1.0, day_case=0.65, ward_los=(0, 3), icu=0.05,
     types=(
        (id=2030, mu=3.535375437, sigma=0.336187744, gamma=0.0),
        (id=2032, mu=3.466075601, sigma=0.659585526, gamma=9.989115873),
        (id=1982, mu=3.956866171, sigma=0.222067648, gamma=5.438123056),
        (id=1987, mu=4.129063261, sigma=0.177422244, gamma=5.540697140),
        (id=2036, mu=2.863509335, sigma=1.140191733, gamma=38.995817110),
        (id=1994, mu=4.907187672, sigma=0.158995627, gamma=5.186048079),
     )),
)

const _ORSCHED_BENCHMARK_LOADS = collect(0.80:0.05:1.20)
const _ORSCHED_BENCHMARK_OR_DAYS = (5, 10, 15, 20, 25, 30, 35, 40)
