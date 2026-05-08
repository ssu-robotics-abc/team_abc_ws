import time
import rclpy
import DR_init
#!/usr/bin/env python3

from pymodbus.client.sync import ModbusTcpClient as ModbusClient


class RG():

    def __init__(self, gripper, ip, port):
        self.client = ModbusClient(
            ip,
            port=port,
            stopbits=1,
            bytesize=8,
            parity='E',
            baudrate=115200,
            timeout=1)
        if gripper not in ['rg2', 'rg6']:
            print("Please specify either rg2 or rg6.")
            return
        self.gripper = gripper  # RG2/6
        if self.gripper == 'rg2':
            self.max_width = 1100
            self.max_force = 400
        elif self.gripper == 'rg6':
            self.max_width = 1600
            self.max_force = 1200
        self.open_connection()

    def open_connection(self):
        """Opens the connection with a gripper."""
        self.client.connect()

    def close_connection(self):
        """Closes the connection with the gripper."""
        self.client.close()

    def get_fingertip_offset(self):
        """Reads the current fingertip offset in 1/10 millimeters.
        Please note that the value is a signed two's complement number.
        """
        result = self.client.read_holding_registers(
            address=258, count=1, unit=65)
        offset_mm = result.registers[0] / 10.0
        return offset_mm

    def get_width(self):
        """Reads current width between gripper fingers in 1/10 millimeters.
        Please note that the width is provided without any fingertip offset,
        as it is measured between the insides of the aluminum fingers.
        """
        result = self.client.read_holding_registers(
            address=267, count=1, unit=65)
        width_mm = result.registers[0] / 10.0
        return width_mm

    def get_status(self):
        """Reads current device status.
        This status field indicates the status of the gripper and its motion.
        It is composed of 7 flags, described in the table below.

        Bit      Name            Description
        0 (LSB): busy            High (1) when a motion is ongoing,
                                  low (0) when not.
                                  The gripper will only accept new commands
                                  when this flag is low.
        1:       grip detected   High (1) when an internal- or
                                  external grip is detected.
        2:       S1 pushed       High (1) when safety switch 1 is pushed.
        3:       S1 trigged      High (1) when safety circuit 1 is activated.
                                  The gripper will not move
                                  while this flag is high;
                                  can only be reset by power cycling.
        4:       S2 pushed       High (1) when safety switch 2 is pushed.
        5:       S2 trigged      High (1) when safety circuit 2 is activated.
                                  The gripper will not move
                                  while this flag is high;
                                  can only be reset by power cycling.
        6:       safety error    High (1) when on power on any of
                                  the safety switch is pushed.
        10-16:   reserved        Not used.
        """
        # address   : register number
        # count     : number of registers to be read
        # unit      : slave device address
        result = self.client.read_holding_registers(
            address=268, count=1, unit=65)
        status = format(result.registers[0], '016b')
        status_list = [0] * 7
        if int(status[-1]):
            print("A motion is ongoing so new commands are not accepted.")
            status_list[0] = 1
        if int(status[-2]):
            print("An internal- or external grip is detected.")
            status_list[1] = 1
        if int(status[-3]):
            print("Safety switch 1 is pushed.")
            status_list[2] = 1
        if int(status[-4]):
            print("Safety circuit 1 is activated so it will not move.")
            status_list[3] = 1
        if int(status[-5]):
            print("Safety switch 2 is pushed.")
            status_list[4] = 1
        if int(status[-6]):
            print("Safety circuit 2 is activated so it will not move.")
            status_list[5] = 1
        if int(status[-7]):
            print("Any of the safety switch is pushed.")
            status_list[6] = 1

        return status_list

    def get_width_with_offset(self):
        """Reads current width between gripper fingers in 1/10 millimeters.
        The set fingertip offset is considered.
        """
        result = self.client.read_holding_registers(
            address=275, count=1, unit=65)
        width_mm = result.registers[0] / 10.0
        return width_mm

    def set_control_mode(self, command):
        """The control field is used to start and stop gripper motion.
        Only one option should be set at a time.
        Please note that the gripper will not start a new motion
        before the one currently being executed is done
        (see busy flag in the Status field).
        The valid flags are:

        1 (0x0001):  grip
                      Start the motion, with the target force and width.
                      Width is calculated without the fingertip offset.
                      Please note that the gripper will ignore this command
                      if the busy flag is set in the status field.
        8 (0x0008):  stop
                      Stop the current motion.
        16 (0x0010): grip_w_offset
                      Same as grip, but width is calculated
                      with the set fingertip offset.
        """
        result = self.client.write_register(
            address=2, value=command, unit=65)

    def set_target_force(self, force_val):
        """Writes the target force to be reached
        when gripping and holding a workpiece.
        It must be provided in 1/10th Newtons.
        The valid range is 0 to 400 for the RG2 and 0 to 1200 for the RG6.
        """
        result = self.client.write_register(
            address=0, value=force_val, unit=65)

    def set_target_width(self, width_val):
        """Writes the target width between
        the finger to be moved to and maintained.
        It must be provided in 1/10th millimeters.
        The valid range is 0 to 1100 for the RG2 and 0 to 1600 for the RG6.
        Please note that the target width should be provided
        corrected for any fingertip offset,
        as it is measured between the insides of the aluminum fingers.
        """
        result = self.client.write_register(
            address=1, value=width_val, unit=65)

    def close_gripper(self, force_val=400):
        """Closes gripper."""
        params = [force_val, 0, 16]
        print("Start closing gripper.")
        result = self.client.write_registers(
            address=0, values=params, unit=65)

    def open_gripper(self, force_val=400):
        """Opens gripper."""
        params = [force_val, self.max_width, 16]
        print("Start opening gripper.")
        result = self.client.write_registers(
            address=0, values=params, unit=65)

    def move_gripper(self, width_val, force_val=400):
        """Moves gripper to the specified width."""
        params = [force_val, width_val, 16]
        print("Start moving gripper.")
        result = self.client.write_registers(
            address=0, values=params, unit=65)
# ======================
# 로봇 및 그리퍼 설정
# ======================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502


class Task3HardwareTester:
    def __init__(self):
        # 1. 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            print("[Init] 그리퍼 연결 성공")
        except Exception as e:
            print(f"[Error] 그리퍼 연결 실패: {e}")

        # 2. 주요 위치 정의 (조인트 좌표 posj 및 직교 좌표 posx)
        # TODO: 실제 캘리브레이션 된 좌표로 변경해야 합니다.
        self.pos_home = posj([0, -20, 120, 0, 15, 90])  # 홈 자세
        
        # 직교 좌표 [x, y, z, rx, ry, rz]
        self.pos_checkout = posx([408.0, 153.0, 342.0, 33.0, 180.0, 100.0])  # 판매대 위치
        self.pos_return = posx([408.0, -153.0, 342.0, 133.0, -172.0, -120.0])   # 반품대 위치
        
        self.safe_z_offset = 100.0  # 이동 시 물체를 긁지 않도록 띄울 Z 높이

    def execute_cycle(self, barcode_id, is_match):
        """Task 3 로직: (파지 상태 유지) 홈 출발 -> 일치 여부 확인 -> 판매대/반품대 이동 후 놓기 -> 홈 복귀"""
        print("\n" + "=" * 50)
        print(f"[Task 3 실행] 물품 ID: {barcode_id} / 일치 여부: {is_match}")
        print("=" * 50)

        # Step 1: 시작 전 홈 위치 이동 (이미 물건을 잡고 있다고 가정하므로 그리퍼 명령 생략)
        print("[Step 1] 물체를 쥐고 있는 상태로 홈 위치 이동/대기")
        movej(self.pos_home, VELOCITY, ACC)
        wait(1.0)

        # Step 2: 조건(is_match)에 따른 목적지 결정
        print(f"[Step 2] 바코드 검증: {barcode_id} -> 일치 상태({is_match})")
        if is_match:
            print("[판매대]로 이동합니다.")
            target_pose = self.pos_checkout
        else:
            print("[반품대]로 이동합니다.")
            target_pose = self.pos_return

        # Step 3: 목적지로 이동 및 Release
        print(f"[Step 3] 목적지 이동 및 물체 놓기")
        # 안전 높이로 목적지 상단까지 이동
        approach_target = posx(target_pose.copy())
        approach_target[2] += self.safe_z_offset
        movel(approach_target, VELOCITY, ACC)
        wait(0.5)
        
        # 하강 후 놓기
        movel(target_pose, VELOCITY, ACC)
        wait(0.5)
        self.gripper.open_gripper()
        time.sleep(1.0)
        
        # 다시 상승
        movel(approach_target, VELOCITY, ACC)
        wait(0.5)

        # Step 4: 홈 위치 복귀
        print("[Step 4] 작업 완료. 빈 그리퍼 상태로 홈 위치 복귀")
        movej(self.pos_home, VELOCITY, ACC)
        wait(1.0)
        
        print("성공적으로 종료됨\n")


def main():
    # Doosan API 사용을 위한 필수 rclpy 초기화 (껍데기 역할)
    rclpy.init()
    node = rclpy.create_node("task3_hardware_test_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # ★ 추가된 부분: 클래스에서도 API를 사용할 수 있도록 전역(global) 선언
    global get_current_posx, movej, movel, wait, posx, posj

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        rclpy.shutdown()
        raise SystemExit(1)

    # 로직 테스터 인스턴스화
    tester = Task3HardwareTester()

    # 테스트 케이스 1: 일치 (판매대 이송)
    #tester.execute_cycle(barcode_id="8801234", is_match=True)
    
    # 테스트 케이스 2: 불일치 (반품대 이송)
    tester.execute_cycle(barcode_id="8809999", is_match=False)

    # 종료 처리
    rclpy.shutdown()


# ======================
# 메인 실행 블록
# ======================
if __name__ == "__main__":
    main()