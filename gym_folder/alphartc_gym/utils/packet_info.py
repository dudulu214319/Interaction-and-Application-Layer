#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None  # int
        self.send_timestamp = None  # int, ms
        self.ssrc = None  # int
        self.padding_length = None  # int, B
        self.header_length = None  # int, B
        self.receive_timestamp = None  # int, ms
        self.payload_size = None  # int, B
        self.bandwidth_prediction = None  # int, bps
        self.first_packet = None  # bool
        self.last_packet = None  # bool

    def __str__(self):
        return (
            f"receive_timestamp: { self.receive_timestamp }ms"
            f", send_timestamp: { self.send_timestamp }ms"
            f", payload_size: { self.payload_size }B"
        )
