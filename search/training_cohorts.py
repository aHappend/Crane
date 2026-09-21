"""Capacity-bounded FW/BW1/recompute-BW2 training reference schedules.

This explicit serial reference reproduces the Figure-6 cohort accounting and
searches uniform checkpoint retention across sub-batches. It does not claim the
paper's entire per-layer checkpoint search space. Backward work is an explicit
two-forward-work analytical approximation; no neural-network weights are trained.
"""
from __future__ import annotations

from dataclasses import asdict
from math import floor

from scheduler.paper_milp import _state_cost_coeffs


class InfeasibleCohortSchedule(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details=details


def run_training_cohorts(workload, config, retained=None):
    candidates=[]
    layers=workload.layers
    # Full internal activations are included, not only inter-block boundaries.
    sample_payload=sum(layer.output_size for layer in layers)
    weights=sum(layer.weight_size for layer in layers)
    input_mb=sum(layer.input_size or 0 for layer in layers)
    edge_mb=sum(parent.output_size for child in layers for parent in child.parents)
    operations=sum(layer.flops for layer in layers)
    for sub_batch in config.candidate_sub_batches:
        if sub_batch<=0 or config.batch_size % sub_batch:
            continue
        q=config.batch_size//sub_batch
        payload=sample_payload*sub_batch
        # Keep one cohort of workspace for gradients. Recompute needs one
        # cohort of activations plus one of workspace after BW1 frees checkpoints.
        max_keep=min(q,max(0,floor(config.dram_capacity/max(payload,1e-30))-1))
        keep=max_keep if retained is None else int(retained)
        if not 0<=keep<=q:
            continue
        discarded=q-keep
        peak=max((keep+1)*payload,2*payload if discarded else 0)
        if peak>config.dram_capacity+1e-8:
            continue
        compute_l=compute_e=0.0
        for layer in layers:
            latency,energy=_state_cost_coeffs([layer.flops*sub_batch],
                [layer.effective_map_dims(sub_batch)],None,None,None,None,1,
                config.num_pes,config.compute_power_per_tile,config.compute_energy_per_op)
            compute_l+=latency[0];compute_e+=energy[0]
        # Conservative checkpoint reference: activations/gradients stream through
        # DRAM. This intentionally exposes recomputation vs memory, not fusion.
        fw_bytes=(input_mb+edge_mb+sample_payload)*sub_batch+weights
        bw_bytes=(edge_mb+2*sample_payload)*sub_batch+weights
        dram_bw=config.dram_bandwidth or config.noc_bandwidth
        def phase(count, scale, volume):
            traffic_l=volume/config.noc_bandwidth+volume/dram_bw
            traffic_e=volume*(config.noc_energy_per_unit+config.dram_energy_per_unit)
            return {'sub_batches':count,'operations':operations*sub_batch*scale*count,
                    'latency_seconds':(compute_l*scale+traffic_l)*count,
                    'energy_joules':(compute_e*scale+traffic_e)*count,
                    'dram_traffic_mb':volume*count}
        fw=phase(q,1,fw_bytes)
        bw1=phase(keep,config.backward_compute_scale,bw_bytes)
        recompute=phase(discarded,1,fw_bytes)
        bw2=phase(discarded,config.backward_compute_scale,bw_bytes)
        phases={'fw':fw,'bw1':bw1,'recompute':recompute,'bw2':bw2}
        latency=sum(p['latency_seconds'] for p in phases.values())
        energy=sum(p['energy_joules'] for p in phases.values())
        candidates.append({'sub_batch':sub_batch,'retained':keep,'discarded':discarded,
            'phases':phases,'latency':latency,'energy':energy,'edp':latency*energy,
            'peak_dram_mb':peak,'payload_mb':payload})
    if not candidates:
        valid=[sb for sb in config.candidate_sub_batches if sb>0 and config.batch_size % sb==0]
        raise InfeasibleCohortSchedule('no cohort schedule fits the activation/checkpoint DRAM budget',
            {'policy':'uniform_cohort_with_gradient_workspace',
             'minimum_required_dram_mb':2*sample_payload*min(valid) if valid else None,
             'available_dram_mb':config.dram_capacity,
             'scope':'infeasible within this reference policy, not a proof for all training schedules'})
    best=min(candidates,key=lambda x:x['edp'])
    sb,k,d=best['sub_batch'],best['retained'],best['discarded']
    q=config.batch_size//sb
    phases=best['phases']
    checks={
        'forward_samples_complete':q*sb==config.batch_size,
        'backward_samples_once':(k+d)*sb==config.batch_size,
        'only_discarded_samples_recomputed':phases['recompute']['sub_batches']==d,
        'dram_capacity':best['peak_dram_mb']<=config.dram_capacity+1e-8,
        'physical_tile_budget':True,  # all operations execute serially on the tile set
    }
    return {
        'status':'completed' if all(checks.values()) else 'validation_failed',
        'cost_model':'analytical_serial_training_uniform_cohort_checkpoints',
        'checks':checks,
        'metrics':{'latency_seconds':best['latency'],'energy_joules':best['energy'],
            'edp_joule_seconds':best['edp'],'best_sub_batch':sb,'peak_dram_mb':best['peak_dram_mb'],
            'peak_sram_mb':0.0,'operations_per_batch':operations*config.batch_size,
            'training_operations':sum(p['operations'] for p in phases.values()),
            'retained_sub_batches':k,'recomputed_sub_batches':d},
        'phases':phases,
        'cohorts':{'fw_indices':[1,q],'bw1_indices':[d+1,q] if k else [],
                   'recompute_bw2_indices':[1,d] if d else []},
        'checkpoint':{'forward_final_met_d':[d]*len(layers),'activation_payload_mb':best['payload_mb'],
                      'capacity_scope':'activations plus one cohort of gradient workspace; static weights/optimizer state excluded'},
        'candidates':candidates,
        'limitations':['uniform cohort checkpoints','serial execution','analytical backward scale',
                       'no optimizer-state capacity accounting','not the full paper training search'],
    }
