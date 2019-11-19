from tensor import Tensor

class Node():
  def __init__(self,
               node_name,
               start_time=0,
               end_time=0):
    self.node_name = node_name

    # Can be delayed due to other operation. such as swap in
    self.logic_time = 0

    # Init by the run_metadata
    self.start_time = start_time
    self.start_rel_time = 0
    self.end_time = end_time  # end_rel_time
    self.exec_time = 0

    self.tmp_time = 0

    self.ref_tensors = []

    self.outputs = dict()

    self.pending_count = -1
    self.fanout_nodes = []

  def AddTime(self, node_stat):
    # Init time info
    all_start_micros = node_stat.all_start_micros
    all_end_rel_micros = node_stat.all_end_rel_micros

    if self.start_time == 0:
      self.start_time = all_start_micros
    else:
      self.start_time = min(self.start_time, all_start_micros)

    self.tmp_time += all_end_rel_micros
    self.exec_time = all_start_micros - self.start_time + all_end_rel_micros
    if self.tmp_time > self.exec_time:
      print("%s: Weird time!" % self.node_name)
      print("%d v.s. %d" % (self.exec_time, self.tmp_time))
      self.exec_time = self.tmp_time


  def InitOutput(self, node_stat):
    # Init tensor info
    for i in node_stat.output:
      i_slot = i.slot
      td = i.tensor_description

      allocation_d = td.allocation_description

      requested_bytes = allocation_d.requested_bytes
      allocated_bytes = allocation_d.allocated_bytes
      allocator_name = allocation_d.allocator_name

      tensor_name = self.node_name+'_'+str(i_slot)
      if self.outputs.__contains__(tensor_name):
        # print("Weird tensor : %s" % tensor_name)
        pass
      else:
        self.outputs[tensor_name] = Tensor(self.node_name, tid=i_slot,
                                           requested_bytes=requested_bytes,
                                           allocator_name=allocator_name,
                                           allocated_bytes=allocated_bytes,
                                           allocated_time=self.start_time)



  def __cmp__(self, other):
    if self.pending_count == other.pending_count:
      return self.start_time > other.start_time

    return self.pending_count > other.pending_count
