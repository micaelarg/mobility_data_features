# data/external_data/location_data.py
# el gdp es de la provincia 
# https://statistics.cepal.org/portal/databank/index.html?lang=es&indicator_id=4952&area_id=2567&members=29190%2C84461%2C84721%2C84722%2C84723%2C84724%2C84725%2C84726%2C84727%2C84728%2C84729%2C84730%2C84731%2C84732%2C84733%2C84734%2C84735%2C84736%2C84737%2C84738%2C84739%2C84740%2C84741%2C84742%2C84743%2C84744%2C140765
cities_data = { 
    'Quito': {
        'coords': (-0.1865944, -78.4305382),
        'population': 1763275,
        'gdp_per_capita': 23448, #provincia 8717.9
        'elevation': 2850
    },
    'Guayaquil': {
        'coords': (-2.1525012, -79.9801008),
        'population': 2650288,
        'gdp_per_capita': 45214, #provincia 14428
        'elevation': 4
    },
    'Cuenca': {
        'coords': (-2.8922693, -79.98938),
        'population': 596101,
        'gdp_per_capita': 13158,  #provincia 5480.2
        'elevation': 2550
    },
    'Manta': {
        'coords': (-0.9683169, -80.7095146),
        'population': 258697,
        'gdp_per_capita': 12622,
        'elevation': 6
    },
    'Ambato': { 
        'coords': (-1.2571436, -78.6216187),
        'population': 380000,
        'gdp_per_capita': 4295.9,
        'elevation': 2577
    },
    'Machala': {
        'coords': (-3.258111, -79.955343),
        'population': 241606,  
        'gdp_per_capita': 5071.4, 
        'elevation': 6
        },
    'Santa Elena': {
        'coords': (-2.2280962,-80.8610702),
        'population': 86450, 
        'gdp_per_capita': 2938.1,  
        'elevation': 40
    },
    'Riobamba': {
        'coords': (-1.6659943,-78.7359595),
        'population': 260882,  
        'gdp_per_capita': 2322.2,  
        'elevation': 3500
    },
    'Santo Domingo de los Tsáchilas': {
        'coords': (-0.3403084,-79.4769862),
        'population': 492969,  
        'gdp_per_capita': 4153.5,  
        'elevation': 800
    },
    # 'Sangolquí': {
    #     'coords': (-0.3357695,-78.4574346),
    #     'population': 96647,  
    #     'gdp_per_capita': 8717.9,
    #     'elevation': 2500
    # },
    'Ibarra': {
        'coords': (0.351663,-78.1435014),
        'population': 131856,  
        'gdp_per_capita': 3055.3,  
        'elevation': 2225
    }
}

commercial_areas = [
    {'coords': (-2.1557518,-79.8927122), 'size': 'large'},  # Mall del Sol
    {'coords': (-0.1759917,-78.4819501), 'size': 'large'},  # Quicentro Shopping
    {'coords': (-2.1250723,-79.9079671), 'size': 'medium'}, # Mall del Rio
    {'coords': (-2.0979007,-79.8793732), 'size': 'medium'}, # Plaza Lagos
    {'coords': (-2.1416104,-79.9117627), 'size': 'large'},  # City Mall
    {'coords': (-0.2527686,-78.5285635), 'size': 'medium'}, # Centro Comercial El Recreo, Quito
    {'coords': (-0.1781061,-78.4787846), 'size': 'medium'}, # Quicentro Shopping, Quito
    {'coords': (-1.2652797,-78.6281705), 'size': 'medium'}, # Mall de los Andes, Ambato
    {'coords': (-0.942805,-80.7348508), 'size': 'medium'}, # Mall del Pacífico, Manta
    {'coords': (-2.8873378,-78.9912676), 'size': 'medium'},   # Multiplaza Miraflores, Cuenca
    {'coords': (0.3491438,-78.1252875), 'size': 'small'},     # Laguna Mall, Ibarra
    {'coords': (-2.1765245,-79.9758568), 'size': 'medium'},   # Policentro, Guayaquil
]

universities = [
    {'coords': (-0.1855501,-78.4688008), 'ranking': 1},  # USFQ
    {'coords': (-0.1854428,-78.4997007), 'ranking': 2},  # EPN
    {'coords': (-2.9004548,-79.01226659), 'ranking': 3},  # U. Cuenca
    {'coords': (-2.4590878,-79.7057418), 'ranking': 4},  # UCSG
    {'coords': (-2.1810801,-79.9012732), 'ranking': 5},  # U. Guayaquil
    {'coords': (-2.1478802,-79.9995883), 'ranking': 6}   # ESPOL
]

tourist_spots = [
    {'coords': (-0.0021418,-78.4557418), 'importance': 'high'},    # Mitad del Mundo
    {'coords': (-2.1819671,-79.8768684), 'importance': 'medium'}, # Las Peñas
    {'coords': (-2.1938946,-79.8901539), 'importance': 'high'},   # Malecón 2000
    {'coords': (-0.220423,-78.5150881), 'importance': 'high'},   # Catedral de Cuenca
    {'coords': (-2.3247621,-81.5163336), 'importance': 'medium'},   # Catedral de Cuenca
    {'coords': (-0.1921688,-78.5220933), 'importance': 'high'},   # Teleférico de Quito
    {'coords': (-2.8479288,-79.3336887), 'importance': 'high'}    # Parque Nacional Cajas
]

geographic_features = {
    'coast_line': [
        (-3.3838, -80.2238),  # cerca de la frontera con Perú
        (-3.1000, -80.1671),
        (-2.9020, -80.1046),
        (-2.7087, -80.0467),
        (-2.5237, -79.9705),
        (-2.3676, -79.9020),
        (-2.1589, -79.8543),
        (-1.9627, -79.8744),
        (-1.8022, -79.9144),
        (-1.6386, -79.9393),
        (-1.4981, -80.0176),
        (-1.2700, -80.0912),
        (-1.0365, -80.1445),
        (-0.8500, -80.2104),
        (-0.6708, -80.2810),
        (-0.4417, -80.3123),
        (-0.2296, -80.3299),
        (-0.0831, -80.3784),
        (0.1626, -80.4163),
        (0.3454, -80.4523),
        (0.5606, -80.4674),
        (0.7878, -80.4954),
        (0.9987, -80.5068),
        (1.2093, -80.5198),
        (1.4016, -80.5481),  # Cerca de Esmeraldas
        (1.5272, -80.5917),
        (1.7414, -80.6356),
        (1.9506, -80.6835),
        (2.1526, -80.7212),
        (2.3016, -80.7482),
        (2.5201, -80.7896),  # cerca de la frontera con Colombia
    ],
    'mountains': [
        {'coords': (-1.9979923,-79.1359219), 'elevation': 6263}, # Chimborazo
        {'coords': (-0.6837326,-78.4474898), 'elevation': 5896}, # Volcán Cotopaxi
        {'coords': (0.0250215,-77.9994663), 'elevation': 5790},  # Volcán Cayambe
        {'coords': (-0.484923,-78.1519713), 'elevation': 5704}, # Volcán Antisana
        {'coords': (-1.4701812,-78.4551052), 'elevation': 5023}, # Volcán Tungurahua
        {'coords': (-0.1708111,-78.6229349), 'elevation': 4776}  # Volcán Pichincha
    ]
}
