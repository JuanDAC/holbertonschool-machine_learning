import tkinter as tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)


class RenderFigure:
    """
    RenderFigure represents a gui application
    that can be used to render any figure
    """

    def __init__(self, **kwargs):
        """
        Construct gui application object, configuration and init mainloop
        """
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.master = tk.Tk()
        self.master.wm_title(self.title)
        self.createFigure()
        self.createToolbar()
        self.addEvents()
        self.create_exit_action()
        self.master.mainloop()

    def createFigure(self):
        """
        Create a figure element in canvas
        """
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.draw()
        self.canvas\
            .get_tk_widget()\
            .pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def createToolbar(self):
        """
        Create a toolbar that handler the figure
        """
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas\
            .get_tk_widget()\
            .pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def addEvents(self):
        """
        Add events to the figure canvas
        """
        self.canvas.mpl_connect("key_press_event", self.on_key_press())

    def on_key_press(self):
        """
        Handle key press events
        """
        def key_press(event):
            """
            Event handler for key press events
            """
            print("you pressed {}".format(event.key))
            key_press_handler(event, self.canvas, self.toolbar)
        return key_press

    def create_exit_action(self):
        """
        Create a action add the exit event
        """
        button = tk.Button(master=self.master, text="Quit",
                           command=self._quit())
        button.pack(side=tk.BOTTOM)

    def _quit(self):
        """
        handler the exit event
        """
        def to_quit():
            """
            Event to clear anf destroy
            """
            self.master.quit()
            self.master.destroy()
        return to_quit
