from ray.rllib.algorithms.callbacks import DefaultCallbacks
from controllables.core import BaseVariable
from controllables.core.tools.records import VariableRecords

class PMVVariable(BaseVariable):
    def __init__(
        self, 
        tdb: BaseVariable,
        tr: BaseVariable,
        rh: BaseVariable,
        metab_rate=1.0, clothing=.5, pmv_limit=.5,
    ):
        self.tdb = tdb
        self.tr = tr
        self.rh = rh
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)
    
    @property
    def value(self):
        res = pytc.models.pmv_ppd(
            tdb=self.tdb.value, 
            tr=self.tr.value, 
            # calculate relative air speed
            vr=pytc.utilities.v_relative(v=0.1, met=self._metab_rate), 
            rh=self.rh.value, 
            met=self._metab_rate, 
            # calculate dynamic clothing
            clo=pytc.utilities.clo_dynamic(clo=self._clothing, met=self._metab_rate),
            limit_inputs=False,
        )['pmv']

        return res


class PlottingCallbacks(DefaultCallbacks):
    def __init__(self, zone_map: dict[str, str] | None = None):
        # 设计思路：
        # 1) 允许从外部传入 zone_map；若未传，则默认用 env.room_agent_ids 或全局 ZONE_MAP
        # 2) 仅在首次 episode 启动时构建 VariableRecords，避免重复绑定
        self.env_records: dict[object, VariableRecords] | None = None
        self.zone_map = zone_map

    def on_episode_start(self, *, episode, worker, **kwargs):
        env: UserEnv = worker.env
        system = env.system

        if self.env_records is None:
            system.add('logging:progress')
            self.env_records = {}

            # 选择映射优先级：env.room_agent_ids > self.zone_map > 全局 ZONE_MAP
            zone_map: dict[str, str] = getattr(env, "room_agent_ids", None) or self.zone_map or ZONE_MAP

            for agent_id in env.agent.observation_space:
                # 跳过不需要的设备代理
                if agent_id in ('CHILLER', 'AHU'):
                    continue

                # ✅ 核心：用 ZONE_MAP 把 agent 别名 -> EnergyPlus 真实分区名 var_key
                try:
                    var_key = zone_map[agent_id]
                except KeyError as e:
                    # 更早暴露问题：如果没映射到，直接报错方便你修表
                    raise KeyError(f"agent_id '{agent_id}' 未在 ZONE_MAP 中，无法绑定输出变量") from e

                # 用 var_key 正确构造 E+ 输出变量引用（**不要再用别名 key**）
                tr = system[OutputVariable.Ref(
                    type='Zone Mean Radiant Temperature',
                    key=var_key,
                )]
                rh = system[OutputVariable.Ref(
                    type='Zone Air Relative Humidity',
                    key=var_key,
                )]

                # 其余变量直接复用观测里已经绑定好的量（这些在 UserEnv 里已按 var_key 绑定好）
                tdb = env.agent.observation[agent_id]['temperature_drybulb']
                pv = env.agent.observation[agent_id]['PV']
                hvac_energy = env.agent.observation[agent_id]['energy_consumption']
                occupancy = env.agent.observation[agent_id]['occupancy']
                # 组合 PMV
                pmv = PMVVariable(tdb=tdb, tr=tr, rh=rh)

                # 建立该 agent 的记录器（VariableRecords 会在 poll() 时采样所有变量）
                self.env_records[agent_id] = VariableRecords({
                    'time'   : system['time'],
                    'reward' : env.agent.reward,
                    'pmv'    : pmv,
                    'occupancy': occupancy,
                    'temp'   : tdb,
                    'elec'   : hvac_energy,
                    'pv'     : pv,
                })

    def on_episode_step(self, *, episode, **kwargs):
        # 每步采样一把
        for _, env_records in self.env_records.items():
            env_records.poll()

    def on_episode_end(self, *, episode, **kwargs):
        # 回合结束落盘
        records_dir = RESULT_DIR / RUN_NAME / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        for agent_ref, env_records in self.env_records.items():
            env_records.dataframe().to_csv(
                records_dir / f"records_{agent_ref}.csv",
                index=False
            )
