"""
Este arquivo utiliza as funcoes do arquivo Seleciona_Pontos.py para
marcar os pontos do contorno automaticamente, salvar em uma imagem os pontos
marcados e salvar as coordenadas dos pontos eu um arquivo texto.
"""

import select_points as s

paciente = 'Matsushita'
plano = 'Coronal'
sequencia = 21
lado = 1
if plano == 'Coronal':
    if lado == 0:
        origem = '{}/{}/{}_L'.format(paciente, plano, sequencia)
    else:
        origem = '{}/{}/{}_R'.format(paciente, plano, sequencia)
else:
    origem = '{}/{}/{}'.format(paciente, plano, sequencia)


def salva_pontos():
    cont = 1
    f = open(origem + "/points.txt", 'a')
    while(cont <= 3):
        nome = '{}/segIM ({}).png'.format(origem, cont)
        salvar = '{}/contour_points ({}).png'.format(origem, cont)
        contorno = s.define_landmarks(nome, salvar, 0)

        lista_pontos = "["
        for i in range(0, len(contorno)):
            aux = contorno[i]
            lista_pontos += str(aux) + ", "
        lista_pontos = lista_pontos[0: -1]
        lista_pontos += "]"

        aux = lista_pontos + '\n'
        f.write(aux)

        cont = cont + 1
    f.close()


salva_pontos()
