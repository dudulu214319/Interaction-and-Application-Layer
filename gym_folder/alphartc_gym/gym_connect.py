#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import json
import logging

__ZMQ_TYPE__ = "ipc://"
__ZMQ_PATH__ = "/tmp/"
__ZMQ_PREFIX__ = __ZMQ_TYPE__ + __ZMQ_PATH__
__GYM_EXIT_FLAG__ = b"Bye"

class GymConnector(object):
    def __init__(self, gym_id = "gym"):
        self.gym_id = gym_id
        self.zmq_ctx = zmq.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.REQ)
        self.zmq_sock.connect(__ZMQ_PREFIX__ + self.gym_id)

    def step(self, bandwidth_bps = int):
        # logging.info(f"Bandwidth sent to simulator: {int(bandwidth_bps)}")
        self.zmq_sock.send_string(str(int(bandwidth_bps)))

        rep = self.zmq_sock.recv()
        # print(rep)
        #rep is the packet list
        if rep == __GYM_EXIT_FLAG__:
            # logging.info("GYM EXIT FLAG")
            return None
        return json.loads(rep)

    def __del__(self):
        self.zmq_sock.disconnect(__ZMQ_PREFIX__ + self.gym_id)
