1) Executar arquivo segmentation.py
   Segmentar 3 imagens de uma sequência:
	Uma no momento máximo de expiração
	Uma no momento médio 
	Uma no momento máximo de inspiração
2) Executa arquivo save_points para salvar os 60 pontos de contorno (25 cada lado do pulmão e 10 na região do diafragma)
Obs. Caso seja na região do coração é preciso expecificar isso no  arquivo select_points.py através do parâmetro 'coracao' (caso sagital), isso fará com que sejam salvos 10 pontos antes de alcançar o coração.
3) Executa o arquivo join e escolhe o momento correto para juntar com a região diafragmática segmentada manualmente ou o arquivo union.py caso esteja na região onde contenha o coração (caso sagital)
4) os pontos podem ser conferidos no arquivo join usando a função check_points
5) Executar o arquivo util/unionMasks.py para unir as silhuetas pulmonares dos pulmões em uma única imagem

python segmentation.py
python save_points.py
python join.py -imgnumber=1 -sequence=2 -save=0 -stage=0

