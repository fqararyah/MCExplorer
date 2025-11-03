import __init__
import utils
import constants
from generic_mapping import GenericMapping
from abc import ABC, abstractmethod


class EnergyComponents():
    def __init__(self, on_chip_ifms_idle_energy_per_second, on_chip_ifms_dynamic_energy,
                 on_chip_ofms_idle_energy_per_second, on_chip_ofms_dynamic_energy,
                 on_chip_weights_idle_energy_per_second, on_chip_weights_dynamic_energy,
                 dsps_energy_idle_per_second, dsps_energy_dynamic,
                 off_chip_accesses_energy):

        self.on_chip_ifms_idle_energy_per_second = on_chip_ifms_idle_energy_per_second
        self.on_chip_ifms_dynamic_energy = on_chip_ifms_dynamic_energy
        self.on_chip_ofms_idle_energy_per_second = on_chip_ofms_idle_energy_per_second
        self.on_chip_ofms_dynamic_energy = on_chip_ofms_dynamic_energy
        self.on_chip_weights_idle_energy_per_second = on_chip_weights_idle_energy_per_second
        self.on_chip_weights_dynamic_energy = on_chip_weights_dynamic_energy
        self.dsps_energy_idle_per_second = dsps_energy_idle_per_second
        self.dsps_energy_dynamic = dsps_energy_dynamic
        self.off_chip_accesses_energy = off_chip_accesses_energy


class SimpleMapping(GenericMapping):

    @abstractmethod
    def calc_on_chip_accesses(self, print_desc=False):
        pass

    def calc_energy_components(self):
        num_ops = sum(utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)) / 2
        used_pes = self.get_used_pes()
        on_chip_weight_accesses, on_chip_ifms_accesses, on_chip_ofms_accesses = self.calc_on_chip_accesses()
        weights_buffer_brams = self.calc_weights_buffer_sz() / constants.BRAM_BLOCK_BYTES
        fms_buffer_brams = self.calc_fms_buffer_sz() / constants.BRAM_BLOCK_BYTES

        bram_idle_power = self.hw_config.bram_idle_pow
        bram_read_power = self.hw_config.bram_r_pow
        bram_write_power = self.hw_config.bram_w_pow

        bram_energy_weights_idle = (bram_idle_power * weights_buffer_brams)
        bram_energy_weights_dynamic = on_chip_weight_accesses * \
            bram_read_power / self.hw_config.frequency

        bram_energy_ifms_idle = (bram_idle_power * fms_buffer_brams / 2)
        bram_energy_ifms_dynamic = on_chip_ifms_accesses * \
            bram_read_power / self.hw_config.frequency

        bram_energy_ofms_idle = (bram_idle_power * fms_buffer_brams / 2)
        bram_energy_ofms_dynamic = on_chip_ofms_accesses * \
            bram_write_power / self.hw_config.frequency

        dsps_energy_idle = self.hw_config.dsp_idle_pow * used_pes
        dsps_energy_dynamic = self.hw_config.dsp_dynamic_pow * num_ops / self.hw_config.frequency

        off_chip_access = self.calc_off_chip_fms_access(
        ) + self.calc_off_chip_weights_access()

        off_chip_accesses_energy = self.hw_config.off_chip_access_energy * off_chip_access
        
        return EnergyComponents(bram_energy_ifms_idle, bram_energy_ifms_dynamic,
                                bram_energy_ofms_idle, bram_energy_ofms_dynamic,
                                bram_energy_weights_idle, bram_energy_weights_dynamic,
                                dsps_energy_idle, dsps_energy_dynamic,
                                off_chip_accesses_energy)

    def calc_energy(self, print_desc=False):
        energy_comps = self.calc_energy_components()
        exec_time = self.calc_exec_time()

        bram_energy_ifms_idle = energy_comps.on_chip_ifms_idle_energy_per_second * exec_time
        bram_energy_ofms_idle = energy_comps.on_chip_ofms_idle_energy_per_second * exec_time
        bram_energy_weights_idle = energy_comps.on_chip_weights_idle_energy_per_second * exec_time

        bram_energy_ifms_accesses = energy_comps.on_chip_ifms_dynamic_energy
        bram_energy_ofms_accesses = energy_comps.on_chip_ofms_dynamic_energy
        bram_energy_weights_accesses = energy_comps.on_chip_weights_dynamic_energy

        off_chip_accesses_energy = energy_comps.off_chip_accesses_energy

        dsps_idle_energy = energy_comps.dsps_energy_idle_per_second * exec_time
        dsps_comp_energy = energy_comps.dsps_energy_dynamic

        total_energy = bram_energy_ifms_idle + bram_energy_ofms_idle + bram_energy_weights_idle + \
            bram_energy_ifms_accesses + bram_energy_ofms_accesses + bram_energy_weights_accesses + \
            off_chip_accesses_energy + dsps_idle_energy + dsps_comp_energy

        if print_desc:
            print('bram_energy_ifms_idle: {}\n'.format(bram_energy_ifms_idle),
                  'bram_energy_ofms_idle: {}\n'.format(bram_energy_ofms_idle),
                  'bram_energy_weights_idle: {}\n'.format(
                      bram_energy_weights_idle),
                  'bram_energy_ifms_accesses: {}\n'.format(
                      bram_energy_ifms_accesses),
                  'bram_energy_ofms_accesses: {}\n'.format(
                      bram_energy_ofms_accesses),
                  'bram_energy_weights_accesses: {}\n'.format(
                      bram_energy_weights_accesses),
                  'off_chip_accesses_energy: {}\n'.format(
                      off_chip_accesses_energy),
                  'dsps_idle_energy: {}\n'.format(dsps_idle_energy),
                  'dsps_comp_energy: {}\n'.format(dsps_comp_energy))

        return total_energy
    

    def get_used_pes(self):
        used_pes = 0
        for engine in self.engines:
            used_pes += engine.get_parallelism()

        return used_pes
