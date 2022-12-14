# Trading System

This trading system is based on machine learning and intends to proceed to the micro alpha factory.

## Beta Version 0.1 (공개)

- 베타 버전은 백테스트 기능만 제공합니다.
- PoC (Proof of Concept) 로 구현한 것이라 path 설정 및 data 유무에 따라 엔진이 작동하지 않을 수 있습니다.

## Beta Version 0.2 (비공개)

- Backtester_GS.py로 무수히 많은 전략을 돌린 후 file 형태로 결과를 적재함과 동시에 시각화 및 요약 기능 추가
- 최종적으로 전략 선택 후 Trader 클래스에 연결시켜 바이낸스 선물 거래를 통해 실전 작동합니다.
- 전략 생성과 실전 거래의 괴리에서 나오는 많은 기술적 괴리 및 오류를 '빠르게' 고치는 것에 집중
    - 이로 인해 코드가 깔끔하지는 않으나 비즈니스적 관점에서 시스템의 가치성을 판단하는 것이 더 중요하다고 봤고, 이를 대략적으로 검증한 이후에 재설계를 생각함.
- 실전 6 개월간 실전에서 실험해보면서 1) 기술적 버그 및 2) 전략적 버그를 다양하게 발견했고, 가능성을 확인함.

- 使用Backtester_GS.py运行无数策略后，结果以文件形式加载，同时增加可视化和汇总功能。
- 最后选择策略后，连接到Trader类，通过币安合约交易实际操作。
- 专注于“快速”修复由于策略创建和实际交易之间的差距而产生的许多技术差距和错误
- 因此，代码不干净，但我认为从业务角度判断系统的价值更重要，在大致验证后，我想到了重新设计。
- 在6个月的实战试验中，发现了各种1）技术BUG和2）战略BUG，并确认了可能性。

## Beta Version 0.3 (진행중)

- 이를 반영함과 동시에 robust한 Trading System을 만들고자 재설계 작업을 진행 중.
- Front-end / Back-end / Engine / DB 로 구별하여 작업 진행 예정
    - B2C 서비스를 제공 수준은 아니나 ‘코드’ 단에서 관리하기 보단 인터페이스를 나누는 것이 좋을 것이라 판단
    - 사고 실험을 통해 가치를 창출할 수 있는 Trading System의 Architecture를 그리는 중
- **Trading/Simulator Engine의 핵심** : **거래비용을 감안한 회전율 X 확률우위**
- 为了反映这一点并同时创建一个强大的交易系统，正在进行重新设计。
- 预定分为Front-end / Back-end / Engine / DB
- 虽然不是提供B2C服务的层面，但判断与其在‘代码’阶段管理，不如划分接口更好
- 通过思想实验绘制出可以创造价值的交易系统架构
- **交易/模拟器引擎的核心**：**换手率 X 考虑交易成本的概率优势**

---
# Trading Research
  WIKI : https://github.com/jaeaehkim/trading_system_beta/wiki

---
# 부록
- Beta Version 0.1 실행을 위해선 Engine.Backtester_GS.py를 run 하면 됨.
- 이론적 기반은 Advances in Financial Machine Learning 1st Edition by Marcos Lopez de Prado 에 기반하고 있으며 해당 책을 베이스로 하여 많은 부분을 변형 및 추가함.
- afml_flow를 통해 Marcos Lopez의 이론들을 코드로 실습해볼 수 있음.


＃ 附录
- 要运行 Beta 版本 0.1，请运行 Engine.Backtester_GS.py。
- 理论基础基于Marcos Lopez de Prado的Advances in Financial Machine Learning 1st Edition，许多部分在本书的基础上进行了修改和补充。
- 您可以通过afml_flow 用代码实践Marcos Lopez 的理论。
