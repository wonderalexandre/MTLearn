# Comparação da Jacobiana implícita em `dat/misc256`

A representação compacta melhorou o backward de 38/39 imagens em CPU e de 39/39 em MPS nesta execução. O ganho mediano sobre a versão original foi de 1,24x em CPU e 2,39x em MPS. Em MPS, o tempo ficou próximo ao da versão por eventos sem ordenação; a maior parte do ganho já aparece nessa primeira etapa.

## Entradas e protocolo

- Data da execução: 2026-09-10. Foram usadas todas as **39 imagens**, todas 256 x 256: **25 em escala de cinza e 14 RGB**.
- Os canais RGB foram preservados e processados separadamente, como na CFP: **67 canais escalares**, dois tipos de árvore (max/min), dois dispositivos (CPU/MPS), **268 casos**, três variantes por caso.
- Nenhuma imagem foi redimensionada ou convertida para cinza. Os arquivos originais e seus hashes estão no [manifesto](input_manifest.csv).
- Ambiente: `macOS-26.6.2-arm64-arm-64bit`, PyTorch `2.10.0`, FP32, quatro threads CPU, seed 71.
- Mesmo protocolo da comparação sintética: três aquecimentos, dez repetições medidas, rotação da ordem das variantes e sincronização MPS antes/depois de cada medição. As repetições, medianas e quartis estão nos [resultados brutos](results.json).
- Os sinais de entrada das operações são resíduos das árvores reais multiplicados por gates sigmoid pseudoaleatórios fixos. O gradiente recebido também é fixado pela seed. Todas as variantes usam os mesmos sinais; não há treinamento ou scorer aprendido nessa medição.
- Foram medidos os operadores forward/backward isolados e, separadamente, a construção da ordem em CPU. Leitura das imagens, construção das árvores, extração nativa dos metadados, atributos, normalização, transferências, autograd e otimização estão fora dos tempos.
- A variante original recebe a ordem pré-calculada, como ocorria na camada. Assim, o ganho do backward na etapa 1 inclui a remoção das conversões de ranks e prefixos redundantes, e não deve ser atribuído apenas à remoção de `argsort` da preparação.

## Agregação

Para cada imagem e variante, somamos as medianas de todos os canais e das duas árvores. Esses valores representam a soma dos custos isolados, **não uma medição conjunta da camada**. Em seguida, calculamos os speedups pareados por imagem e sua mediana sobre as 39 imagens, dando o mesmo peso a imagens RGB e em cinza. Uma razão acima de 1 indica melhora. As medianas das razões podem diferir da razão entre as medianas dos tempos.

## Ganhos por operação

| Dispositivo | Operação | Original / eventos | Original / compacta | Eventos / compacta | Imagens com compacta mais rápida que original |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | forward | 1.003x | 1.057x | 1.052x | 37/39 |
| CPU | backward | 1.194x | 1.237x | 1.043x | 38/39 |
| MPS | forward | 0.997x | 0.998x | 0.999x | 16/39 |
| MPS | backward | 2.400x | 2.394x | 0.994x | 39/39 |

Diferenças pequenas, sobretudo no forward em MPS, não estabelecem superioridade de desempenho. Os números de imagens com melhora contam apenas a ordem observada das medianas; não são testes de significância.

## Construção da ordem em CPU

| Variante | Mediana do custo agregado por imagem (ms) | Ganho pareado mediano sobre original |
| --- | ---: | ---: |
| original_argsort | 0.663 | 1.00x |
| events_linear | 0.236 | 2.55x |
| compact_linear | 0.056 | 10.91x |

A versão compacta também reduz o comprimento dos vetores de eventos/prefixos de aproximadamente `2T` para `T+1`. Os vetores de índices `pre` e `post` continuam tendo `T` elementos cada. Isso não significa dividir pela metade a memória total da camada; não foi medido pico de memória nesta execução.

## Resultado por imagem: backward

Tempos em milissegundos, somando medianas dos canais e de max-tree/min-tree.

| Imagem | Canais | CPU original | CPU eventos | CPU compacta | MPS original | MPS eventos | MPS compacta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.1.01.png | 3 | 1.110 | 0.967 | 0.949 | 5.441 | 2.264 | 2.289 |
| 4.1.02.png | 3 | 1.230 | 1.052 | 1.028 | 5.512 | 2.483 | 2.499 |
| 4.1.03.png | 3 | 1.258 | 1.091 | 1.039 | 5.544 | 2.736 | 2.808 |
| 4.1.04.png | 3 | 1.135 | 0.993 | 0.951 | 5.516 | 2.318 | 2.378 |
| 4.1.05.png | 3 | 1.294 | 1.068 | 1.028 | 5.821 | 2.518 | 2.479 |
| 4.1.06.png | 3 | 1.306 | 1.071 | 1.022 | 5.657 | 2.314 | 2.292 |
| 4.1.07.png | 3 | 1.171 | 1.125 | 1.119 | 5.480 | 2.814 | 2.810 |
| 4.1.08.png | 3 | 1.174 | 1.096 | 1.041 | 5.109 | 2.534 | 2.566 |
| 4.2.01.png | 3 | 1.149 | 1.007 | 0.944 | 5.481 | 2.330 | 2.385 |
| 4.2.03.png | 3 | 1.371 | 1.047 | 1.015 | 5.796 | 2.253 | 2.288 |
| 4.2.05.png | 3 | 1.189 | 1.000 | 0.971 | 5.463 | 2.339 | 2.366 |
| 4.2.06.png | 3 | 1.260 | 1.011 | 0.974 | 5.689 | 2.276 | 2.274 |
| 4.2.07.png | 3 | 1.162 | 0.985 | 0.956 | 5.506 | 2.295 | 2.262 |
| 5.1.09.png | 1 | 0.432 | 0.363 | 0.338 | 1.868 | 0.761 | 0.722 |
| 5.1.10.png | 1 | 0.455 | 0.379 | 0.356 | 1.923 | 0.761 | 0.743 |
| 5.1.11.png | 1 | 0.408 | 0.355 | 0.340 | 1.802 | 0.807 | 0.835 |
| 5.1.12.png | 1 | 0.380 | 0.326 | 0.332 | 1.829 | 0.886 | 0.862 |
| 5.1.13.png | 1 | 0.550 | 0.536 | 0.534 | 2.183 | 1.295 | 1.304 |
| 5.1.14.png | 1 | 0.416 | 0.339 | 0.322 | 1.827 | 0.751 | 0.738 |
| 5.2.08.png | 1 | 0.435 | 0.340 | 0.326 | 1.806 | 0.753 | 0.752 |
| 5.2.09.png | 1 | 0.438 | 0.345 | 0.329 | 1.931 | 0.743 | 0.766 |
| 5.2.10.png | 1 | 0.414 | 0.343 | 0.335 | 1.899 | 0.756 | 0.736 |
| 5.3.01.png | 1 | 0.394 | 0.325 | 0.322 | 1.956 | 0.760 | 0.745 |
| 5.3.02.png | 1 | 0.397 | 0.327 | 0.316 | 1.856 | 0.736 | 0.738 |
| 7.1.01.png | 1 | 0.407 | 0.337 | 0.322 | 1.921 | 0.783 | 0.818 |
| 7.1.02.png | 1 | 0.400 | 0.352 | 0.351 | 1.780 | 0.938 | 0.934 |
| 7.1.03.png | 1 | 0.397 | 0.334 | 0.312 | 1.830 | 0.751 | 0.753 |
| 7.1.04.png | 1 | 0.452 | 0.379 | 0.349 | 1.904 | 0.755 | 0.772 |
| 7.1.05.png | 1 | 0.411 | 0.342 | 0.319 | 1.861 | 0.770 | 0.796 |
| 7.1.06.png | 1 | 0.404 | 0.333 | 0.318 | 1.975 | 0.793 | 0.802 |
| 7.1.07.png | 1 | 0.424 | 0.337 | 0.318 | 1.894 | 0.746 | 0.782 |
| 7.1.08.png | 1 | 0.411 | 0.357 | 0.339 | 1.813 | 0.887 | 0.826 |
| 7.1.09.png | 1 | 0.418 | 0.335 | 0.328 | 2.018 | 0.758 | 0.784 |
| 7.1.10.png | 1 | 0.401 | 0.328 | 0.314 | 1.891 | 0.765 | 0.787 |
| 7.2.01.png | 1 | 0.378 | 0.329 | 0.315 | 1.885 | 0.931 | 0.965 |
| boat.512.png | 1 | 0.408 | 0.341 | 0.335 | 1.846 | 0.793 | 0.771 |
| gray21.512.png | 1 | 0.479 | 0.540 | 0.540 | 1.776 | 1.003 | 1.042 |
| house.png | 3 | 1.318 | 1.092 | 1.058 | 5.668 | 2.362 | 2.321 |
| ruler.512.png | 1 | 0.535 | 0.480 | 0.481 | 2.087 | 1.037 | 0.993 |

A exceção em CPU foi `gray21.512.png`: a compacta levou 0.540 ms contra 0.479 ms da original, aproximadamente 12.6% a mais. Isso é consistente com o caso de poucos nós e muitos pixels já observado nas imagens sintéticas; não há ganho universal em CPU.

## Verificação numérica

Todas as variantes produziram valores finitos. Maior erro absoluto em relação ao produto por intervalos em FP64, usando os mesmos sinais:

| Variante | Forward | Backward |
| --- | ---: | ---: |
| original_cached | 1.220e-04 | 2.026e-08 |
| events_scan | 1.220e-04 | 2.492e-08 |
| compact_scan | 1.132e-04 | 2.026e-08 |

Os erros observados são pequenos na escala de intensidade de 0 a 255 usada pelos resíduos, mas a comparação não exige igualdade bit a bit. Os testes independentes de gradiente e de suporte das subárvores foram executados na implementação anterior deste mesmo refactor; aqui foi conferida a cobertura completa da pasta e a consistência das medições, sem alterar a implementação da CFP.

## Reproduzir

```bash
PYTHONPATH=mtlearn/python:build/cfp-compact-jacobian/mtlearn/bindings \
python scripts/benchmarks/cfp_jacobian_intervals.py \
  --stage compact --image-dir dat/misc256 \
  --warmups 3 --repeats 10 --devices cpu mps \
  --output /tmp/misc256-comparison.json
```

O ambiente local tinha uma instalação editable apontando para outro checkout. A execução usou o mesmo bootstrap temporário descrito no relatório anterior para carregar este código e sua build. Os caminhos efetivos estão registrados em `results.json`.

Arquivos: [resultados completos](results.json), [resumo](summary.json), [tabela por imagem](per_image.csv), [manifesto das entradas](input_manifest.csv).
