# CFP cache — resultado da etapa P0

Data: 2026-09-09. **P0 concluída; P1 ainda não iniciada.**

A referência anterior ao refatoramento está congelada e validada. Três imagens de 2.748 × 2.748 pixels completaram preparação e forward/backward em CPU; a imagem 0 também completou o consumo em MPS. A biblioteca e os algoritmos nativos permaneceram intactos, conforme os hashes dos fontes registrados antes e depois da execução.

O principal resultado para o planejamento é o tamanho do armazenamento: **aproximadamente 477 GiB para os 2.041 pares**, usando a média das três amostras e persistindo apenas os dados brutos atuais. O cache legado de um modelo projetaria aproximadamente **785 GiB de tensores**. Armazenar em SSD resolve a retenção em RAM, mas ainda exige uma cota compatível com esse volume.

## 1. Entregas e procedência

- [Referências congeladas](../../../mtlearn/tests/python/fixtures/cfp_cache_p0/README.md): imagens pequenas, divisão treino/teste, configurações, contratos, estatísticas, checkpoints, saídas, perdas e gradientes; integridade por SHA-256.
- [Testes de referência](../../../mtlearn/tests/python/test_cfp_cache_baseline.py): 61 casos aprovados e três falhas esperadas da implementação atual.
- [Medidor isolado](../../../scripts/benchmarks/cfp_cache_p0.py): import explícito deste checkout, medição por imagem, monitor externo e limites de execução.
- [Ambiente da referência](environment.json), [inventário do dataset](dataset-inventory.json), [medições em CSV](measurements.csv), [consolidação em JSON](summary.json) e [validação](validation.json).
- [Resultados completos dos testes](validation-tests.xml) e registros por execução: [CPU 0](runs/cpu_0000/result.json), [CPU 1000](runs/cpu_1000/result.json), [CPU 2040](runs/cpu_2040/result.json), [MPS 0](runs/mps_0000/result.json). Cada diretório inclui ambiente, tempos, log e amostras do monitor.

O checkout observado foi `fd29ec8b39ff7bdb1cb30d4b89a02b9b37a2224d`, com alterações preexistentes fora da CFP registradas no ambiente. O backend `external/mmcfilters` estava em `70cb54efbe394863bb85c7917f996150c9bb283f` (`v4.2.0`). Não foi usado o pacote de outro checkout apontado pela instalação editável da máquina.

## 2. Referência de correção

As fixtures usam cinco imagens sintéticas determinísticas de 6 × 7 pixels: três para ajustar estatísticas e duas para verificar saídas e gradientes. Seed 42; entrada, atributos e parâmetros em `float32`; momentos estatísticos em CPU `float64` e contagens inteiras. Os parâmetros de referência são não nulos e os testes verificam gradientes finitos.

| Árvore | Scorers | Normalização | Atributos |
| --- | --- | --- | --- |
| Max-tree | Linear e MLP | `none`, min-max, z-score e clipped z-score | AREA/GRAY_HEIGHT; ALL no clipped z-score |
| Min-tree | Linear e MLP | Clipped z-score | AREA/GRAY_HEIGHT |
| Tree of shapes | Linear e MLP | Clipped z-score | AREA/GRAY_HEIGHT |

São 12 configurações. Para cada uma, os testes comparam execução direta e com cache com a mesma referência congelada, tanto em CPU quanto em MPS, verificando saídas, perda e gradientes dos parâmetros. A avaliação não pode alterar as estatísticas do treino. Também há testes de restauração de checkpoints/estatísticas em CPU e ajuste das estatísticas contra os valores congelados.

Tolerâncias iniciais: CPU `rtol=1e-5`, `atol=1e-6`; CPU/MPS `rtol=1e-4`, `atol=1e-5`; estatísticas `rtol=1e-10`, `atol=1e-12`, com chaves e valores não tensoriais exatos. As fixtures ocupam aproximadamente 466 KiB, sem imagens privadas do dataset real.

**Validação: 242 testes aprovados, três falhas esperadas, zero falhas inesperadas.** Inclui 61 novos casos de referência e 181 testes existentes, dos quais 19 são checagens numéricas de gradientes. Houve um aviso de `TypedStorage` durante a apresentação de uma falha esperada pelo pytest.

### Limitação preexistente encontrada no MPS

`load_checkpoint(device="mps")`, `load_state_dict` em uma camada MPS e `load_stats` nessa camada tentam transferir momentos `float64` ao MPS. Os três caminhos falham com `TypeError`. Carregar o arquivo inicialmente em CPU não basta para corrigir o segundo caminho: o desserializador da camada volta a transferir os momentos ao dispositivo.

Os três casos ficam como `xfail(strict=True)`: sua correção futura produz XPASS e exige atualizar os testes. **A biblioteca não foi corrigida na P0.** Os testes numéricos MPS usam o caminho atualmente funcional: ajustar estatísticas sobre as três imagens de treino e copiar somente os parâmetros congelados para a camada MPS. Portanto, a equivalência matemática em MPS passou; a compatibilidade de restauração MPS continua sendo uma pendência concreta da P1. Os checkpoints congelados foram restaurados integralmente em CPU.

## 3. Protocolo de recursos

O inventário por nomes e cabeçalhos PNG confirmou 2.048 inputs e 2.041 máscaras; faltam as máscaras dos IDs 2041–2047. Todos os arquivos têm 2.748 × 2.748 pixels, um canal e oito bits. Foram medidas as imagens 0, 1000 e 2040, sempre com sua máscara e sem redimensionamento.

A configuração corresponde à do notebook: max-tree, 63 atributos de `AttributeGroup.ALL`, clipped z-score, sharpness 1, clamp 12, inicialização próxima da identidade `p0=0.995`, linear com 64 parâmetros e MLP `[8]`/tanh com 521 parâmetros. Cada execução ajustou as estatísticas somente sobre sua própria imagem para medir o custo; isso **não é o ajuste estatístico do experimento completo nem uma avaliação de segmentação**. Os testes científicos pequenos, separados, usam treino e teste distintos.

O processo executa leitura, conversão, construção da árvore, extração dos tensores/atributos, estatísticas e normalização pelo construtor de cache legado em CPU. Mantém uma única amostra. Serializa uma cópia lógica dos dados brutos, com tensores CPU e metadados primitivos, em arquivo temporário no SSD externo; mede tamanho e abertura por mmap, verificando mapa pixel–nó e todos os atributos. O arquivo temporário é removido ao terminar. Depois mede forward, perda e backward dos dois scorers, um de cada vez.

No ensaio MPS, a preparação continua em CPU e o payload legado já preparado é transferido explicitamente ao acelerador. A cópia CPU permanece viva. Isso isola o consumo do payload no MPS; não reproduz a construção integral do cache legado diretamente em MPS, nem o pico do notebook com dois caches completos. A normalização desse ensaio ocorre na preparação CPU. A nova normalização por lote no acelerador será medida quando existir.

A instrumentação é externa à biblioteca. Há um processo de medição por vez, uma thread interna por biblioteca e monitoramento de RSS do trabalhador e seus descendentes a cada 100 ms. Limites usados: interrupção acima de 4 GiB de RSS, abaixo de 1 GiB de memória disponível no sistema ou após 600 segundos. Nenhum limite foi acionado. O supervisor observa limites; não impõe uma cota rígida ao alocador. Picos muito curtos podem escapar da amostragem.

## 4. Medições observadas

Valores binários: MiB = 2²⁰ bytes; GiB = 2³⁰ bytes. Uma observação por imagem/dispositivo, sem repetição estatística.

| Imagem / consumo | Nós | Dados brutos (MiB) | Cache legado (MiB) | Pico RSS preparação (GiB) | Pico RSS total (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / CPU | 699.690 | 255,13 | 423,28 | 1,28 | 1,48 |
| 1000 / CPU | 627.731 | 234,81 | 385,67 | 1,25 | 1,41 |
| 2040 / CPU | 601.906 | 227,52 | 372,18 | 1,22 | 1,45 |
| 0 / MPS | 699.690 | 255,13 | 423,28 | 1,24 | 1,24 |

O pico MPS reportado na última coluna é somente RSS observado. Nos finais das etapas, o alocador MPS registrou até **0,86 GiB ativos** e **1,10 GiB no driver**. Essas medidas não são somáveis automaticamente ao RSS, pois há memória unificada, reservas do driver e possível sobreposição. A menor memória disponível do sistema no ensaio MPS foi **2,14 GiB**. A preparação CPU atingiu seu maior RSS antes da serialização; o maior RSS total dos ensaios CPU ocorreu na perda/backward.

| Etapa (segundos) | CPU 0 | CPU 1000 | CPU 2040 | Ensaio MPS 0 |
| --- | ---: | ---: | ---: | ---: |
| Leitura e decodificação | 0,124 | 0,058 | 0,055 | 0,054 |
| Construção da árvore | 0,976 | 0,776 | 0,705 | 0,820 |
| Extração dos atributos | 20,385 | 18,834 | 19,163 | 19,005 |
| Atualização estatística | 0,068 | 0,065 | 0,058 | 0,068 |
| Normalização total | 0,151 | 0,129 | 0,116 | 0,143 |
| Preparação CPU total | 21,865 | 20,058 | 20,283 | 20,315 |
| Serialização bruta no SSD | 0,368 | 0,326 | 0,346 | 0,332 |
| Abertura mmap + verificação, leitura aquecida | 0,036 | 0,036 | 0,032 | 0,035 |
| Transferência do payload linear ao MPS | — | — | — | 0,223 |
| Linear: forward / backward | 0,104 / 0,118 | 0,087 / 0,130 | 0,084 / 0,119 | 0,093 / 0,376 |
| Transferência do payload MLP ao MPS | — | — | — | 0,191 |
| MLP: forward / backward | 0,119 / 0,123 | 0,124 / 0,130 | 0,107 / 0,119 | 0,073 / 0,159 |

A preparação total inclui as etapas internas; não somar essa linha às anteriores. A normalização inclui a aplicação inicial e a atualização final do cache legado. Os tempos de forward/backward excluem a perda, registrada separadamente no CSV. O ensaio linear MPS inclui custos iniciais de execução; não se deve concluir superioridade de um scorer ou dispositivo a partir dessas observações únicas.

A extração dos atributos representa aproximadamente **93,9% do tempo de preparação CPU agregado**. Atualizar as estatísticas levou cerca de 58–68 ms por imagem. Essa observação favorece combinar resumos durante a preparação e reutilizá-los; não demonstra ainda o custo da futura implementação incremental. Repetir toda a preparação a cada época continuaria caro: uma extrapolação simples da média dá aproximadamente **11,8 horas por passagem sequencial** no conjunto completo, sem incluir todo o restante do experimento.

A gravação foi medida com `torch.save`, sem fsync; a releitura ocorreu logo após a escrita e pode usar o cache do sistema. São custos observados de serialização e leitura aquecida, não medições de durabilidade ou vazão sustentada do SSD. Não foi medido um armazenamento novo: o formato usado é uma sonda dos tensores legados brutos.

## 5. Orçamento inicial sugerido

O ambiente da referência foi registrado às **15:18:54 -03:00**, em Apple M4, dez CPUs lógicas, **16 GiB de memória unificada**, Python 3.12.3, PyTorch 2.10.0, NumPy 2.4.4 e OpenCV 4.13.0. Havia aproximadamente **3,14 GiB disponíveis**, **11,51 GiB de swap ocupado** e **574,96 GiB livres no SSD externo**. Cada ensaio tem sua própria observação datada. Esses números descrevem a sessão, não capacidades garantidas ou padrões da biblioteca.

A extensão usada foi compilada em diretório isolado, modo Release, Clang 21, backend CHECKED e uma tarefa de compilação. Seu SHA-256 é `c84f7e6cb8b9a6677324de94fd9a8372eedb16e3980ed931379b2caaf5a2ff6c`. O ambiente conserva os argumentos/flags completos.

### Memória e concorrência

Para o próximo piloto neste equipamento, usar **um trabalhador, uma tarefa em andamento, nenhuma fila de imagens grandes e nenhuma retenção em RAM por padrão**. Planejar inicialmente **2 GiB para uma tarefa CPU**: é aproximadamente 1,35 vez o maior pico total observado. Isso é uma reserva inicial a validar com mais imagens, não uma garantia para qualquer entrada. Manter a preparação separada do treinamento enquanto o espaço disponível permanecer próximo dos valores desta sessão.

Uma opção explícita de cache RAM de **256 MiB** é suficiente para cerca de uma entrada bruta das amostras medidas, mas aumenta a memória retida além dos temporários. Não a somar a um orçamento já ocupado por treinamento sem nova medição. Imagens maiores devem poder ser consumidas sem admissão nesse cache.

Manter o monitor conservador do piloto em 4 GiB de RSS e mínimo de 1 GiB disponível. Esses limites são do diagnóstico, não uma proposta de reserva de 4 GiB por worker. Dois workers exigiriam pelo menos cerca de 4 GiB apenas para tarefas CPU, além do coordenador, filas e eventual treinamento; **a P0 não validou dois processos simultâneos**. O experimento de concorrência continua na P4.

### SSD

| Projeção para 2.041 imagens | Mínimo amostrado × N | Média amostrada × N | Máximo amostrado × N |
| --- | ---: | ---: | ---: |
| Arquivos de dados brutos | 453,53 GiB | 476,71 GiB | 508,55 GiB |
| Tensores do cache legado, um modelo | 741,81 GiB | 784,73 GiB | 843,67 GiB |

São extrapolações de três imagens espaçadas, **não intervalos de confiança nem limites superiores do dataset**. Os dados brutos atuais usam `8P + 296N` bytes nas amostras, sendo P pixels e N nós: `8P + 44N` em topologia/mapas e `252N` em 63 atributos float32. A serialização acrescentou aproximadamente 18–20 KiB por imagem. A camada antiga acrescenta outros `252N` de atributos normalizados. Os valores não incluem imagens, máscaras, modelos, filas ou temporários. Dois caches legados separados projetariam cerca de 1.569 GiB de tensores pela média.

Para persistir o conjunto completo, usar como **planejamento preliminar** o máximo amostrado extrapolado mais 20%: **610,25 GiB de cota**, e manter pelo menos **100 GiB livres fora dessa cota**. Seriam necessários aproximadamente **710 GiB livres antes da preparação**; o SSD desta sessão tinha cerca de 575 GiB. A margem de 20% cobre planejamento, metadados e variação não observada; não prova que todas as imagens caibam.

Para pilotos no espaço atual, uma **cota explícita de 450 GiB** deixaria cerca de 125 GiB livres, mas seria inferior até à menor extrapolação amostrada. Portanto, ela **não permite prometer persistência integral**. A P3 deve interromper novas gravações ao atingir a cota e permitir retomada, conforme o plano; não remover entradas válidas silenciosamente. Persistir tudo exige liberar/adicionar espaço e ampliar a amostragem antes de iniciar. Preparação sob demanda continua sendo a alternativa funcional, com o custo de reconstrução nos misses.

## 6. Reprodução

Executar a partir da raiz deste checkout com o mesmo Python usado para compilar a extensão. Os comandos abaixo usam o medidor, que verifica os imports reais e isola a instalação editável apenas dentro do processo. O exemplo conserva um diretório de build separado do build habitual.

```bash
cmake -S . -B build/cfp-cache-p0-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMTLEARN_BUILD_PYTHON=ON -DMTLEARN_WITH_TORCH=ON \
  -DMTLEARN_ENABLE_ASSERTS=OFF -DMTLEARN_BUILD_TESTS=OFF \
  -DMTLEARN_ENABLE_EMBED=OFF \
  -DPYTHON_EXECUTABLE=/opt/anaconda3/bin/python \
  -DPython3_EXECUTABLE=/opt/anaconda3/bin/python \
  -DPython3_ROOT_DIR=/opt/anaconda3 \
  -DCMAKE_PREFIX_PATH="$(/opt/anaconda3/bin/python -c 'import torch, pybind11; print(torch.utils.cmake_prefix_path + ";" + pybind11.get_cmake_dir())')"
cmake --build build/cfp-cache-p0-release --parallel 1

/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py test -- -q \
  mtlearn/tests/python/test_cfp_cache_baseline.py \
  mtlearn/tests/python/test_cfp_cache.py \
  mtlearn/tests/python/test_cfp_components.py \
  mtlearn/tests/python/test_cfp_deterministic.py \
  mtlearn/tests/python/test_cfp_validation.py \
  mtlearn/tests/python/test_gradchecks.py

/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py supervise \
  --image-id 0 --device cpu --output build/cfp-cache-p0-repeat/cpu_0000
```

Repetir o último comando sequencialmente com IDs 1000 e 2040, em diretórios novos, e depois ID 0 com `--device mps`. O supervisor recusa sobrescrever um diretório existente. Em outra máquina, ajustar o Python, caminhos de build/dataset e limites ao ambiente observado. O driver admite `--build-dir` antes do subcomando.

As referências congeladas não devem ser regeneradas para fazer uma mudança passar. O modo `freeze` permanece disponível para auditoria/reprodução em outro diretório usando a revisão e configuração registradas; o manifesto original define os valores esperados. Não houve treinamento completo, cache do dataset inteiro, benchmark concorrente ou mudança de comportamento da CFP nesta etapa.

## 7. Encaminhamento para P1

A P1 pode partir das fixtures e do perfil de recursos acima: separar preparação bruta e estatísticas, conservar momentos em CPU, preparar constantes congeladas e manter os checkpoints existentes. Incluir a restauração MPS nas correções de compatibilidade, substituindo seus três xfails por verificações positivas quando funcionar. A redução de memória e a eliminação de trabalho repetido serão demonstradas contra esta referência, sem alterar os valores esperados.
