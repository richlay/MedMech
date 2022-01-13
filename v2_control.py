import argparse
import threading
import time
import pickle as pk
import serial
from filter import BandpassFilter1D, NotchFilter1D
from processing import MeanShift1D, Detrend1D, Resample1D, Normalize1D
from feature_extraction import *


import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

#######################################################################################################################################
runThread = 1

# Control table address
ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION      = 30
ADDR_MX_PRESENT_POSITION   = 36

# Protocol version
PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL1_ID                      = 1
DXL2_ID                      = 2
DXL3_ID                      = 3
DXL4_ID                      = 4                 # Dynamixel ID : 1
BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600
DEVICENAME                  = 'COM3'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque

DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

#######################################################################################################################################

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

#######################################################################################################################################

# Enable Dynamixel Torque

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel ID1 has been successfully connected")

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel ID2 has been successfully connected")

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL3_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel ID3 has been successfully connected")

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL4_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel ID4 has been successfully connected")
#######################################################################################################################################

# define each move_DXL

def move_DXL1(dxl1_goal_position):
    # Write id1 goal position id1
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_MX_GOAL_POSITION, dxl1_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    while 1:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL1_ID, ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL1_ID, dxl1_goal_position, dxl_present_position))

        if not abs(dxl1_goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
            break

def move_DXL2(dxl2_goal_position):
    # Write goal position id2
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_MX_GOAL_POSITION, dxl2_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    while 1:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL2_ID, ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL2_ID, dxl2_goal_position, dxl_present_position))

        if not abs(dxl2_goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
            break

def move_DXL3(dxl3_goal_position):
    # Write goal position id3
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL3_ID, ADDR_MX_GOAL_POSITION, dxl3_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    while 1:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL3_ID, ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL3_ID, dxl3_goal_position, dxl_present_position))

        if not abs(dxl3_goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
            break

def move_DXL4(dxl4_goal_position):
    # Write goal position id4
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL4_ID, ADDR_MX_GOAL_POSITION, dxl4_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    while 1:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL4_ID, ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL4_ID, dxl4_goal_position, dxl_present_position))

        if not abs(dxl4_goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
            break

#######################################################################################################################################


# process signal of each channel
def process_signal1d(x, raw_fs=1000, low_fs=10, high_fs=100, notch_fs=50, Q=0.5, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    # mean-correct signal
    x_processed = MeanShift1D.apply(x)
    # filtering noise
    x_processed = BandpassFilter1D.apply(x_processed, low_fs, high_fs, order=4, fs=raw_fs)
    x_processed = NotchFilter1D.apply(x_processed, notch_fs, Q=Q, fs=raw_fs)
    # detrend
    x_processed = Detrend1D.apply(x_processed, detrend_type='locreg', window_size=window_size, step_size=step_size)
    # resample
    x_processed = Resample1D.apply(x_processed, raw_fs, target_fs)
    # rectify
    x_processed = abs(x_processed)
    # normalize
    x_processed = Normalize1D.apply(x_processed, norm_type='min_max')
    return x_processed

# process multi-channel signal
def process_signalnd(x, raw_fs=1000, low_fs=10, high_fs=100, notch_fs=50, Q=0.5, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    num_channels = x.shape[1]
    x_processed = np.array([])
    for i in range(num_channels):
        # process each channel
        channel_processed = process_signal1d(x[:, i], raw_fs, low_fs, high_fs, notch_fs, Q, window_size, step_size, target_fs)
        channel_processed = np.expand_dims(channel_processed, axis=1)
        if i == 0:
            x_processed = channel_processed
            continue
        x_processed = np.hstack((x_processed, channel_processed))
    return x_processed


# The class to connect the electrodes through serial
class SerialPort:
    def __init__(self, port='COM4', baud=9600, cls=None, pca=None, num_channels=1, interval=1000, timeout=0.1):
        super(SerialPort, self).__init__()
        self.port = serial.Serial(port, baud)
        self.signal = None
        self.interval = interval
        self.cls = cls
        self.num_channels = num_channels
        self.timeout = timeout
        self.feature_window_size = 10   # Please modify as your setting
        self.concat = True   # Please change as your setting
        self.avg_pool = True # Please change as your setting
        self.pca = pca
        self.action = '9'


    def serial_open(self):
        if not self.port.isOpen():
            self.port.open()

    def serial_close(self):
        self.port.close()

    def serial_send(self):
        while runThread == 1:
            print('Send action...')
            print(self.action)
            time.sleep(self.timeout)
            if self.action == '[1]':
                print("Action 1 (rise-arm)")
                # define your '1' action
                # rise-arm
                move_DXL2(512)
                move_DXL1(512)
                move_DXL3(512)
                move_DXL4(512)
                time.sleep(0.5)
                move_DXL4(512)
                move_DXL3(512)
                move_DXL1(512)
                move_DXL2(250)
                time.sleep(1)
                self.action = '9'
            elif self.action == '[2]':
                print("Action 2 (twist-arm)")
                # define your '2' action
                # twist-arm
                move_DXL2(300)
                move_DXL1(700)
                move_DXL3(512)
                move_DXL4(512)
                time.sleep(0.5)
                move_DXL4(512)
                move_DXL3(512)
                move_DXL1(512)
                move_DXL2(250)
                time.sleep(1)
                self.action = '9'
            elif self.action == '[3]':
                print("Action 3 (twist-towel)")
                # define your '3' action
                # twist-towel
                move_DXL2(300)
                move_DXL1(600)
                move_DXL3(400)
                move_DXL4(300)
                time.sleep(0.5)
                move_DXL4(512)
                move_DXL3(512)
                move_DXL1(512)
                move_DXL2(250)
                time.sleep(1)
                self.action = '9'

    def serial_read(self):
        print('Receiving signal...')
        self.action = '[1]'
        qq = 1
        while runThread == 1:
            values = []
            # read signal from serial
            for i in range(self.interval):
                string = self.port.readline().decode('utf-8').rstrip()  # Read and decode a byte string
                print(string)
                values.extend([float(value) for value in string.split(' ')])
            # reshape signal
            signal = np.reshape(np.array(values), (self.interval, 1), order='C')
            # process signal
            # please change parameters as your settings
            signal_processed = process_signalnd(signal, raw_fs=1000, low_fs=10, high_fs=100, notch_fs=50, Q=0.5, window_size=300, step_size=50, target_fs=512)
            # extract, transpose and flatten feature vectors
            # change your feature as your setting
            peak = MaxPeak.apply(signal_processed, self.feature_window_size).T.flatten()
            mean = Mean.apply(signal_processed, self.feature_window_size).T.flatten()
            var = Variance.apply(signal_processed, self.feature_window_size).T.flatten()
            std = StandardDeviation.apply(signal_processed, self.feature_window_size).T.flatten()
            skew = Skewness.apply(signal_processed, self.feature_window_size).T.flatten()
            kurt = Kurtosis.apply(signal_processed, self.feature_window_size).T.flatten()
            rms = RootMeanSquare.apply(signal_processed, self.feature_window_size).T.flatten()
            if self.concat:
                feature = np.hstack([peak, mean, var, std, skew, kurt, rms])
                feature = np.expand_dims(feature, axis=0)
            else:
                feature = np.vstack([peak, mean, var, std, skew, kurt, rms])
                if self.avg_pool:
                    # average pooling
                    feature = feature.mean(axis=0)
                else:
                    # max pooling
                    feature = feature.max(axis=0)
                feature = np.expand_dims(feature, axis=0)
            if self.pca:
                feature = self.pca.transform(feature)
            y_preds = self.cls.predict(feature)
            self.action = str(y_preds)
            #if qq==1:
            #    self.action = '1'
            #    qq = 0

            


#######################################################################################################################################



if __name__ == '__main__':
    
    # robot to default position
    move_DXL2(250)
    move_DXL1(512)
    move_DXL3(512)
    move_DXL4(512)

    # define classifier
    cls = pk.load(open('svc.pkl', 'rb'))
    # define pre-processing pca
    pca = pk.load(open('pca.pkl', 'rb'))
    # Setup serial line
    mserial = SerialPort('COM4', 9600, cls, pca, 1, 580, 0.1)   #565no 570,585yes 590no
    t1 = threading.Thread(target=mserial.serial_read)
    t1.start()
    print("Thread serial_read started")
    # try:
    #     while True:
    #         mserial.serial_send()
    # except KeyboardInterrupt:
    #     print('Press Ctrl-C to terminate while statement')
    t2 = threading.Thread(target=mserial.serial_send)
    t2.start()
    print("Thread serial_send started")

    # 主執行緒繼續執行自己的工作
    while 1:
        if getch() == chr(0x1b):
            print("ESC pressed")
            runThread = 0
            break

    t1.join()
    print("Thread serial_read closed")
    t2.join()
    print("Thread serial_send closed")

    mserial.serial_close()
    print("Serial closed")



#######################################################################################################################################

# Disable Dynamixel 1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque 1 disabled")

# Disable Dynamixel 2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque 2 disabled")

# Disable Dynamixel 3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL3_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque 3 disabled")

# Disable Dynamixel 4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL4_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque 4 disabled")

#######################################################################################################################################

# Close port
portHandler.closePort()
print("Motor port disabled")
print("Program terminated")
