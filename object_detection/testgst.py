import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

Gst.init(None)
a = 0

#rtspsrc的srcpad是随机衬垫，这里使用回调函数来连接衬垫。
def on_pad_added( src, pad, des):
    vpad = des.get_static_pad("sink")
    pad.link(vpad)

pipe = Gst.Pipeline.new("test")
src = Gst.ElementFactory.make("rtspsrc", "src")
depay = Gst.ElementFactory.make("rtph264depay", "depay")
queuev1 = Gst.ElementFactory.make("queue2", "queue")
LATENCY = Gst.Pipeline.get_latency()
#src.connect("pad-added", on_pad_added)
src.connect("pad-added", on_pad_added, queuev1)
#过滤
vfilter = Gst.ElementFactory.make("capsfilter", "flt")
caps = Gst.Caps.from_string("video/x-h264, width=(int)1280, height=(int)720")
vfilter.set_property("caps", caps)

conv = Gst.ElementFactory.make("videoconvert", "conv")
sink = Gst.ElementFactory.make("xvimagesink", "sink")

decodebin = Gst.ElementFactory.make("avdec_h264", "decodea")

rstp = 'rtsp://192.168.99.1/media/stream2'
src.set_property("location", rstp)
pipe.add(src)
pipe.add(depay)

pipe.add(queuev1)
#pipe.add(vfilter)
#pipe.add(conv)

pipe.add(decodebin)
pipe.add(sink)
queuev1.link(depay)
depay.link(decodebin)
#depay.link(vfilter)
#vfilter.link(decodebin)
decodebin.link(sink)
#decodebin.link(conv)
#conv.link(sink)

pipe.set_state(Gst.State.PLAYING)

mainloop = GLib.MainLoop()
mainloop.run()

