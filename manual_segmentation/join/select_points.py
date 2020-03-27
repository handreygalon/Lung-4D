from PIL import Image
import math as mt
import numpy as np

branco = (255, 255, 255)
vermelho = (255, 0, 0)
verde = (0, 255, 0)
azul = (0, 0, 255)
l = h = 255


def monta_lista_pontos(nome, lado_pulmao):
    img1 = Image.open(nome).convert('RGB')
    lista_pontos = []
    for i in range(0, 255):  # eixo x
        for j in range(0, l):  # eixo y
            cor = img1.getpixel((i, j))
            # if cor == verde:
            if cor == branco:
                lista_pontos.append((i, j))
    return lista_pontos


def topo(nome, lista_pontos):
    # acha o p01
    lista_pontos = sorted(lista_pontos, key=lambda lista_pontos: lista_pontos[1])
    return lista_pontos[0]


def canto(nome, lado, lista_pontos):
    # acha cantos
    if lado == 0:  # esquerdo
        canto_x = 0
        canto_y = 255
    elif lado == 1:  # direito
        canto_x = 255
        canto_y = 255
    else:
        print("Informe os parametros corretamente. 0 para lado esquerdo ou 1 para lado direito.")

    lista_ponto_dist = []
    for i in range(0, len(lista_pontos)):
        ponto = lista_pontos[i]
        ponto_em_x = ponto[0]
        ponto_em_y = ponto[1]
        dist_canto = mt.sqrt((ponto_em_x - canto_x)**2 + (ponto_em_y - canto_y)**2)
        lista_ponto_dist.append((ponto, dist_canto))
    ordena_por_dist = sorted(lista_ponto_dist, key=lambda lista_ponto_dist: lista_ponto_dist[1])
    ponto_menor_dist = ordena_por_dist[0]

    # print(ponto_menor_dist[0])
    return ponto_menor_dist[0]


def crescimento(p01, nome):
    # pega o contorno comecando de p01 no sentido anti-horario
    ultimo = (0, 0)
    img1 = Image.open(nome).convert('RGB')
    contorno = [p01]
    img1.putpixel(p01, azul)
    ag = 1
    while (ag != 0):
        ag = 0
        for j in range(0, h):  # eixo y
            for i in range(0, l):  # eixo x
                cor = img1.getpixel((i, j))
                if cor == azul and i < 255 and j < 255 and i > 0 and j > 0:
                    img1.putpixel((i, j), vermelho)
                    esquerda = (i - 1, j)
                    direita = (i + 1, j)
                    cima = (i, j - 1)
                    baixo = (i, j + 1)
                    d_esq_cima = (i - 1, j - 1)
                    d_esq_baixo = (i - 1, j + 1)
                    d_dir_cima = (i + 1, j - 1)
                    d_dir_baixo = (i + 1, j + 1)

                    if (i, j) == p01:
                        if img1.getpixel(baixo) == branco:
                            img1.putpixel(baixo, verde)
                            contorno.append(baixo)
                            ultimo = baixo
                            ag = 1
                        if img1.getpixel(esquerda) == branco:
                            img1.putpixel(esquerda, verde)
                            contorno.append(esquerda)
                            ultimo = esquerda
                            ag = 1
                        if img1.getpixel(d_esq_baixo) == branco:
                            img1.putpixel(d_esq_baixo, verde)
                            contorno.append(d_esq_baixo)
                            ultimo = d_esq_baixo
                            ag = 1
                        if img1.getpixel(cima) == branco:
                            img1.putpixel(cima, vermelho)
                        if img1.getpixel(d_esq_cima) == branco:
                            img1.putpixel(d_esq_cima, vermelho)
                        if img1.getpixel(direita) == branco:
                            img1.putpixel(direita, vermelho)
                        if img1.getpixel(d_dir_cima) == branco:
                            img1.putpixel(d_dir_cima, vermelho)
                        if img1.getpixel(d_dir_baixo) == branco:
                            img1.putpixel(d_dir_baixo, vermelho)
                    else:
                        if img1.getpixel(esquerda) == branco:
                            img1.putpixel(esquerda, verde)
                            contorno.append(esquerda)
                            ultimo = esquerda
                            ag = 1
                        if img1.getpixel(direita) == branco:
                            img1.putpixel(direita, verde)
                            contorno.append(direita)
                            ultimo = direita
                            ag = 1
                        if img1.getpixel(baixo) == branco:
                            img1.putpixel(baixo, verde)
                            contorno.append(baixo)
                            ultimo = baixo
                            ag = 1
                        if img1.getpixel(cima) == branco:
                            img1.putpixel(cima, verde)
                            contorno.append(cima)
                            ultimo = cima
                            ag = 1
                        if img1.getpixel(d_esq_baixo) == branco:
                            img1.putpixel(d_esq_baixo, verde)
                            contorno.append(d_esq_baixo)
                            ultimo = d_esq_baixo
                            ag = 1
                        if img1.getpixel(d_esq_cima) == branco:
                            img1.putpixel(d_esq_cima, verde)
                            contorno.append(d_esq_cima)
                            ultimo = d_esq_cima
                            ag = 1
                        if img1.getpixel(d_dir_cima) == branco:
                            img1.putpixel(d_dir_cima, verde)
                            contorno.append(d_dir_cima)
                            ultimo = d_dir_cima
                            ag = 1
                        if img1.getpixel(d_dir_baixo) == branco:
                            img1.putpixel(d_dir_baixo, verde)
                            contorno.append(d_dir_baixo)
                            ultimo = d_dir_baixo
                            ag = 1

                        if ag == 0:  # nao achou nenhum branco na vizinhanca
                            esquerda2 = (esquerda[0] - 1, esquerda[1])
                            if img1.getpixel(esquerda2) == branco:
                                img1.putpixel(esquerda, verde)
                                contorno.append(esquerda)
                                ultimo = esquerda
                                ag = 1
                            direita2 = (direita[0] + 1, direita[1])
                            if img1.getpixel(direita2) == branco:
                                img1.putpixel(direita, verde)
                                contorno.append(direita)
                                ultimo = direita
                                ag = 1
                            baixo2 = (baixo[0], baixo[1] + 1)
                            if img1.getpixel(baixo2) == branco:
                                img1.putpixel(baixo, verde)
                                contorno.append(baixo)
                                ultimo = baixo
                                ag = 1
                            cima2 = (cima[0], cima[1] - 1)
                            if img1.getpixel(cima2) == branco:
                                img1.putpixel(cima, verde)
                                contorno.append(cima)
                                ultimo = cima
                                ag = 1
                            d_esq_baixo2 = (d_esq_baixo[0] - 1, d_esq_baixo[1] + 1)
                            if img1.getpixel(d_esq_baixo2) == branco:
                                img1.putpixel(d_esq_baixo, verde)
                                contorno.append(d_esq_baixo)
                                ultimo = d_esq_baixo
                                ag = 1
                            d_esq_cima2 = (d_esq_cima[0] - 1, d_esq_cima[1] - 1)
                            if img1.getpixel(d_esq_cima2) == branco:
                                img1.putpixel(d_esq_cima, verde)
                                contorno.append(d_esq_cima)
                                ultimo = d_esq_cima
                                ag = 1
                            d_dir_cima2 = (d_dir_cima[0] + 1, d_dir_cima[1] - 1)
                            if img1.getpixel(d_dir_cima2) == branco:
                                img1.putpixel(d_dir_cima, verde)
                                contorno.append(d_dir_cima)
                                ultimo = d_dir_cima
                                ag = 1
                            d_dir_baixo2 = (d_dir_baixo[0] + 1, d_dir_baixo[1] + 1)
                            if img1.getpixel(d_dir_baixo2) == branco:
                                img1.putpixel(d_dir_baixo, verde)
                                contorno.append(d_dir_baixo)
                                ultimo = d_dir_baixo
                                ag = 1

                    img1.putpixel(ultimo, azul)
        img1.save("contorno_novo.png")
    return contorno


def secundarios(p01, p26, p36, contorno):
    # divide em 3 segmentos (p01-p21, p21-p31, p31-p01)
    ind_p01 = 0
    ind_p26 = 0
    ind_p36 = 0

    p26cima = p26 + (0, -1)
    p26baixo = p26 + (0, -1)
    p26esq = p26 + (-1, 0)
    p26dir = p26 + (11, 0)

    p36cima = p36 + (0, -1)
    p36baixo = p36 + (0, -1)
    p36esq = p36 + (-1, 0)
    p36dir = p36 + (11, 0)

    # print("Contorno: {}\n".format(len(contorno)))
    for i in range(0, len(contorno)):
        if contorno[i] == p26 or contorno[i] == p26cima or contorno[i] == p26baixo or contorno[i] == p26esq or contorno[i] == p26dir:
            ind_p26 = i
        if contorno[i] == p36 or contorno[i] == p36cima or contorno[i] == p36baixo or contorno[i] == p36esq or contorno[i] == p36dir:
            ind_p36 = i
    # print("Indice 1: {}".format(ind_p01))
    # print("Indice 2: {}".format(ind_p26))
    # print("Indice 3: {}\n".format(ind_p36))
    seg1 = contorno[ind_p01: ind_p26]
    seg2 = contorno[ind_p26: ind_p36]
    seg3 = contorno[ind_p36: len(contorno)]
    tam_seg1 = len(seg1)
    tam_seg2 = len(seg2)
    tam_seg3 = len(seg3)
    # print("Tam. seg.: {}, {}, {}\n".format(tam_seg1, tam_seg2, tam_seg3))
    # print("*" * 5)

    intervalo_seg1 = tam_seg1 // 25
    resto1 = tam_seg1 % 25
    lista1 = [0] * 24
    for i in range(1, resto1):
        lista1[i] = 1
    # print("Info seg 1: {}, {}".format(intervalo_seg1, resto1))

    intervalo_seg2 = tam_seg2 // 10
    resto2 = tam_seg2 % 10
    lista2 = [0] * 9
    for i in range(1, resto2):
        lista2[i] = 1
    # print("Info seg 2: {}, {}".format(intervalo_seg2, resto2))

    intervalo_seg3 = tam_seg3 // 25
    resto3 = tam_seg3 % 25
    lista3 = [0] * 24
    for i in range(1, resto3):
        lista3[i] = 1
    # print("Info seg 3: {}, {}".format(intervalo_seg3, resto3))

    ind = [0] * 60
    # define pontos principais
    ind[0] = ind_p01
    ind[25] = ind_p26
    ind[35] = ind_p36
    # print("Pontos principais: {}, {}, {}\n".format(ind[0], ind[25], ind[35]))

    # define indices dos pontos secundarios
    j = 0
    ind_anterior = ind[0]
    for i in range(1, 25):
        ind[i] = ind_anterior + intervalo_seg1 + lista1[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 1: {}".format(ind_anterior))

    j = 0
    ind_anterior = ind[25]
    for i in range(26, 35):
        ind[i] = ind_anterior + intervalo_seg2 + lista2[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 2: {}".format(ind_anterior))

    j = 0
    ind_anterior = ind[35]
    for i in range(36, 60):
        ind[i] = ind_anterior + intervalo_seg3 + lista3[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 3: {}".format(ind_anterior))

    # monta lista de pontos
    landmarks = []
    for i in range(0, 60):
        aux = ind[i]
        landmarks.append(contorno[aux])

    # print(len(landmarks))
    return landmarks


def secundarios_coracao(p01, p11, p36, contorno):
    # divide em 3 segmentos (p01-p21, p21-p31, p31-p01)
    ind_p01 = 0
    ind_p11 = 0
    ind_p36 = 0

    p11cima = p11 + (0, -1)
    p11baixo = p11 + (0, -1)
    p11esq = p11 + (-1, 0)
    p11dir = p11 + (11, 0)

    p36cima = p36 + (0, -1)
    p36baixo = p36 + (0, -1)
    p36esq = p36 + (-1, 0)
    p36dir = p36 + (11, 0)

    # print("Contorno: {}\n".format(len(contorno)))
    for i in range(0, len(contorno)):
        if contorno[i] == p11 or contorno[i] == p11cima or contorno[i] == p11baixo or contorno[i] == p11esq or contorno[i] == p11dir:
            ind_p11 = i
        if contorno[i] == p36 or contorno[i] == p36cima or contorno[i] == p36baixo or contorno[i] == p36esq or contorno[i] == p36dir:
            ind_p36 = i
    # print("Indice 1: {}".format(ind_p01))
    # print("Indice 2: {}".format(ind_p11))
    # print("Indice 3: {}\n".format(ind_p36))
    seg1 = contorno[ind_p01: ind_p11]
    seg2 = contorno[ind_p11: ind_p36]
    seg3 = contorno[ind_p36: len(contorno)]
    tam_seg1 = len(seg1)
    tam_seg2 = len(seg2)
    tam_seg3 = len(seg3)

    intervalo_seg1 = tam_seg1 // 10
    resto1 = tam_seg1 % 10
    lista1 = [0] * 9
    for i in range(1, resto1):
        lista1[i] = 1
    # print("Info seg 1: {}, {}".format(intervalo_seg1, resto1))

    intervalo_seg2 = tam_seg2 // 25
    resto2 = tam_seg2 % 25
    lista2 = [0] * 24
    for i in range(1, resto2):
        lista2[i] = 1
    # print("Info seg 2: {}, {}".format(intervalo_seg2, resto2))

    intervalo_seg3 = tam_seg3 // 25
    resto3 = tam_seg3 % 25
    lista3 = [0] * 24
    for i in range(1, resto3):
        lista3[i] = 1
    # print("Info seg 3: {}, {}".format(intervalo_seg3, resto3))

    ind = [0] * 60
    # define pontos principais
    ind[0] = ind_p01
    ind[10] = ind_p11
    ind[35] = ind_p36
    # print("Pontos principais: {}, {}, {}\n".format(ind[0], ind[25], ind[35]))

    # define indices dos pontos secundarios
    j = 0
    ind_anterior = ind[0]
    for i in range(1, 10):
        ind[i] = ind_anterior + intervalo_seg1 + lista1[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 1: {}".format(ind_anterior))

    j = 0
    ind_anterior = ind[10]
    for i in range(11, 35):
        ind[i] = ind_anterior + intervalo_seg2 + lista2[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 2: {}".format(ind_anterior))

    j = 0
    ind_anterior = ind[35]
    for i in range(36, 60):
        ind[i] = ind_anterior + intervalo_seg3 + lista3[j]
        ind_anterior = ind[i]
        j += 1
    # print("Indice pt. sec. 3: {}".format(ind_anterior))

    # monta lista de pontos
    landmarks = []
    for i in range(0, 60):
        aux = ind[i]
        landmarks.append(contorno[aux])

    # print(len(landmarks))
    return landmarks


def desenha(nome, salvar, landmarks, coracao):
    img1 = Image.open(nome).convert('RGB')
    for i in range(0, 60):
        if coracao:
            if i == 0 or i == 10 or i == 35:
                cor = verde
            else:
                cor = vermelho
        else:
            if i == 0 or i == 25 or i == 35:
                cor = verde
            else:
                cor = vermelho
        img1.putpixel(landmarks[i], cor)

    img1.save(salvar)


def define_landmarks(nome, salvar, lado_pulmao):
    lista_pontos = monta_lista_pontos(nome, lado_pulmao)
    p01 = topo(nome, lista_pontos)
    print("Topo: {}".format(p01))
    p26 = canto(nome, 0, lista_pontos)
    print("Canto esquerdo: {}".format(p26))
    p36 = canto(nome, 1, lista_pontos)
    print("Canto direito: {}\n".format(p36))
    contorno = crescimento(p01, nome)
    coracao = False
    if coracao:
        print("Heart!")
        landmarks = secundarios_coracao(p01, p26, p36, contorno)
    else:
        print("No heart!")
        landmarks = secundarios(p01, p26, p36, contorno)
    desenha(nome, salvar, landmarks, coracao)

    return landmarks
