import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
GObject.threads_init()
import logging

logging.basicConfig()

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)


class VideoPlayer(object):
    '''
    Simple video player
    '''

    source_file = None

    def __init__(self, **kwargs):
        self.loop = GObject.MainLoop()

        if kwargs.get('src'):
            self.source_file = kwargs.get('src')



        self.__setup()

    def run(self):
        self.loop.run()

    def stop(self):
        self.loop.quit()

    def __setup(self):
        _log.info('Setting up VideoPlayer...')
        self.__setup_pipeline()
        _log.info('Set up')

    def __setup_pipeline(self):

        self.pipeline = Gst.Pipeline('video-player-pipeline')
        self.filesrc = Gst.ElementFactory.make("rtspsrc")

        self.filesrc.set_property('location', self.source_file)
        self.pipeline.add(self.filesrc)

        # Demuxer
        self.decoder = Gst.ElementFactory.make('decodebin')
        self.decoder.connect('pad-added', self.__on_decoded_pad)
        self.pipeline.add(self.decoder)

        # Video elements
        self.videoqueue = Gst.ElementFactory.make('queue', 'videoqueue')
        self.pipeline.add(self.videoqueue)

        self.autovideoconvert = Gst.ElementFactory.make('autovideoconvert')
        self.pipeline.add(self.autovideoconvert)

        self.appsink = Gst.ElementFactory.make('appsink')
        self.pipeline.add(self.appsink)

        self.appsink.set_property("emit-signals", True)
        handler_id = self.appsink.connect("new-sample", self.__on_new_sample)

        # Link source and demuxer
        linkres = Gst.element_link_many(
            self.filesrc,
            self.decoder)

        if not linkres:
            _log.error('Could not link source & demuxer elements!\n{0}'.format(
                linkres))

        linkres = Gst.element_link_many(
            self.audioqueue,
            self.audioconvert,
            self.autoaudiosink)

        if not linkres:
            _log.error('Could not link audio elements!\n{0}'.format(
                linkres))

        linkres = Gst.element_link_many(
            self.videoqueue,
            self.progressreport,
            self.autovideoconvert,
            self.autovideosink)

        if not linkres:
            _log.error('Could not link video elements!\n{0}'.format(
                linkres))

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message', self.__on_message)

        #self.pipeline.set_state(gst.STATE_PLAYING)


    def __on_new_sample(self, app_sink):
        sample = app_sink.pull_sample()
        caps = sample.get_caps()

        # Extract the width and height info from the sample's caps
        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")

        # Get the actual data
        buffer = sample.get_buffer()
        # Get read access to the buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Could not map buffer data!")

        numpy_frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data)

        # Clean up the buffer mapping
        buffer.unmap(map_info)
        return numpy_frame


    def __on_decoded_pad(self, pad, data):
        _log.debug('on_decoded_pad: {0}'.format(pad))
        pad.link(self.videoqueue.get_pad('sink'))

    def __on_message(self, bus, message):
        _log.debug(' - MESSAGE: {0}'.format(message))

if __name__ == '__main__':
    player = VideoPlayer(
        src='rtsp://192.168.99.1/media/stream2')

    player.run()