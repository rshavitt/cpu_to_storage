from torch import backends
from nixl._api import nixl_agent, nixl_agent_config
import os
import numpy as np
import time
     

def nixl_write_blocks(block_size, buffer, blocks_indices, file_names):
    """
    Writes discrete blocks from a CPU buffer to files as fast as possible.
    """
    start = time.perf_counter()

    conf = nixl_agent_config(
        enable_prog_thread=True,
        backends=["POSIX"]
    )
    agent = nixl_agent(agent_name="NIXL_Writer", nixl_conf=conf, instantiate_all=False)
    # 1. Memory Registration
    # Must register the CPU buffer to enable zero-copy paths .
    np_buf = buffer.numpy()
    buffer_addr = np_buf.ctypes.data
    
    # Create registration descriptor list and register with agent
    # reg_list = agent.get_reg_descs([(buffer_addr, buffer_len, 0, "")], mem_type="DRAM")
    # reg_handle = agent.register_memory(reg_list, backends=["POSIX"])

    # 2. Build Xfer Descriptor Lists
    # We batch all blocks into one request to minimize syscall overhead.
    local_descs_data = [] # CPU side
    remote_descs_data = [] # File side
    open_fds = []
    temp_files = []  # Track (temp_file, final_file) pairs for renaming

    try:
        for i, (idx, fname) in enumerate(zip(blocks_indices, file_names)):
            # Write to temporary file in /dev/shm first
            temp_fname = f"/dev/shm/temp_block_{idx}.bin"
            temp_files.append((temp_fname, fname))
            fd = os.open(temp_fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            open_fds.append(fd)

            # Local: DRAM block. Format: (addr, len, devID, metadata_handle)
            block_addr = buffer_addr + (idx * block_size)
            local_descs_data.append((block_addr, block_size, 0, ""))

            # Remote: File target. Format: (offset, len, fd, metadata)
            # For FILE mem_type, addr is the offset and devID is the fd .
            remote_descs_data.append((0, block_size, fd, ""))

        # Convert raw lists to NIXL-internal descriptor lists
        local_desc_3 = [t[:-1] for t in local_descs_data]
        local_reg_dl = agent.get_reg_descs(local_descs_data, mem_type="DRAM")
        local_xfer_dl = agent.get_xfer_descs(local_desc_3, mem_type="DRAM")
        remote_desc_3 = [t[:-1] for t in remote_descs_data]
        remote_dl = agent.get_xfer_descs(remote_desc_3, mem_type="FILE")

        reg_handle = agent.register_memory(local_reg_dl)
        file_handle = agent.register_memory(remote_descs_data, mem_type="FILE", backends=["POSIX"])
        file_desc = file_handle.trim()
        # 3. Initialize and Start Transfer
        # Uses loopback (agent's own name) for local storage transfers.
        xfer_handle = agent.initialize_xfer(
            operation="WRITE",
            local_descs=local_xfer_dl,
            remote_descs=file_desc,
            remote_agent="NIXL_Writer",
            backends=["POSIX"]
        )
        
        # Start the asynchronous data movement
        agent.transfer(xfer_handle)

        # 4. Busy Polling (Fastest completion detection)
        # Removing time.sleep() ensures processing resumes the nanosecond I/O ends.
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError("NIXL Transfer Failed")

        # 5. Close file descriptors before renaming
        for fd in open_fds:
            os.close(fd)
        open_fds = []  # Clear the list so finally block doesn't try to close again

        # 6. Rename temporary files to final names atomically
        for temp_fname, final_fname in temp_files:
            os.rename(temp_fname, final_fname)

    finally:
        end = time.perf_counter()

        # 7. Fast Cleanup
        for fd in open_fds:
            os.close(fd)
        if 'xfer_handle' in locals():
            agent.release_xfer_handle(xfer_handle)
        # agent.deregister_memory(reg_handle)

    return (end-start)

def nixl_read_blocks(block_size, buffer,
                           block_indices: list, file_names: list):
    """
    Reads discrete blocks from files into a CPU buffer as fast as possible.
    """
    start = time.perf_counter()

    conf = nixl_agent_config(
        enable_prog_thread=True,
        backends=["POSIX"]
    )
    agent = nixl_agent(agent_name="NIXL_Reader", nixl_conf=conf, instantiate_all=False)

    # 1. Memory Registration
    np_buf = buffer.numpy()
    buffer_addr = np_buf.ctypes.data

    # 2. Build Xfer Descriptor Lists
    local_descs_data = []
    remote_descs_data = []
    open_fds = []

    try:
        for idx, fname in zip(block_indices, file_names):
            fd = os.open(fname, os.O_RDONLY)
            open_fds.append(fd)

            # Local: Destination block in CPU buffer
            block_addr = buffer_addr + (idx * block_size)
            local_descs_data.append((block_addr, block_size, 0, ""))

            # Remote: Source file
            remote_descs_data.append((0, block_size, fd, ""))

        # Convert raw lists to NIXL-internal descriptor lists
        local_desc_3 = [t[:-1] for t in local_descs_data]
        local_reg_dl = agent.get_reg_descs(local_descs_data, mem_type="DRAM")
        local_xfer_dl = agent.get_xfer_descs(local_desc_3, mem_type="DRAM")
        remote_desc_3 = [t[:-1] for t in remote_descs_data]
        remote_dl = agent.get_xfer_descs(remote_desc_3, mem_type="FILE")

        reg_handle = agent.register_memory(local_reg_dl)
        file_handle = agent.register_memory(remote_descs_data, mem_type="FILE", backends=["POSIX"])
        file_desc = file_handle.trim()
        
        # 3. Initialize and Start Transfer
        xfer_handle = agent.initialize_xfer(
            operation="READ",
            local_descs=local_xfer_dl,
            remote_descs=file_desc,
            remote_agent="NIXL_Reader",
            backends=["POSIX"]
        )
        
        agent.transfer(xfer_handle)

        # 4. Busy Polling
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError("NIXL Transfer Failed")

    finally:
        end: float = time.perf_counter()

        for fd in open_fds:
            os.close(fd)
        if 'xfer_handle' in locals():
            agent.release_xfer_handle(xfer_handle)
        # agent.deregister_memory(reg_handle)

    return (end-start)