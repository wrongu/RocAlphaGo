import BaseHTTPServer
import cgi
import logging
import os
import time

HOST_NAME = 'localhost'
PORT_NUMBER = 9000

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

    def do_GET(self):
        self.path = "..\wgo.player-2.0\go.html"

        logging.debug('GET %s' % (self.path))

        # Parse out the arguments.
        # The arguments follow a '?' in the URL. Here is an example:
        #   http://example.com?arg1=val1
        args = {}
        idx = self.path.find('?')

        if idx >= 0:
            rpath = self.path[:idx]
            args = cgi.parse_qs(self.path[idx+1:])
        else:
            rpath = self.path

        # Print out logging information about the path and args.
        if 'content-type' in self.headers:
            ctype, _ = cgi.parse_header(self.headers['content-type'])
            logging.debug('TYPE %s' % (ctype))

        logging.debug('PATH %s' % (rpath))
        logging.debug('ARGS %d' % (len(args)))

        if len(args):
            i = 0
            for key in sorted(args):
                logging.debug('ARG[%d] %s=%s' % (i, key, args[key]))
                i += 1

        # Check to see whether the file is stored locally,
        # if it is, display it.
        # There is special handling for http://127.0.0.1/info. That URL
        # displays some internal information.
        if self.path == '/info' or self.path == '/info/':
            self.send_response(200)  # OK
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.info()
        else:
            # Get the file path.
            path = rpath
            dirpath = None
            logging.debug('FILE %s' % (path))

            # If it is a directory look for index.html
            # or process it directly if there are 3
            # trailing slashed.
            if rpath[-3:] == '///':
                dirpath = path
            elif os.path.exists(path) and os.path.isdir(path):
                dirpath = path  # the directory portion
                index_files = ['/index.html', '/index.htm', ]
                for index_file in index_files:
                    tmppath = path + index_file
                    if os.path.exists(tmppath):
                        path = tmppath
                        break

            # Allow the user to type "///" at the end to see the
            # directory listing.
            if os.path.exists(path) and os.path.isfile(path):
                # This is valid file, send it as the response
                # after determining whether it is a type that
                # the server recognizes.
                _, ext = os.path.splitext(path)
                ext = ext.lower()
                content_type = {
                    '.css': 'text/css',
                    '.gif': 'image/gif',
                    '.htm': 'text/html',
                    '.html': 'text/html',
                    '.jpeg': 'image/jpeg',
                    '.jpg': 'image/jpg',
                    '.js': 'text/javascript',
                    '.png': 'image/png',
                    '.text': 'text/plain',
                    '.txt': 'text/plain',
                }

                # If it is a known extension, set the correct
                # content type in the response.
                if ext in content_type:
                    self.send_response(200)  # OK
                    self.send_header('Content-type', content_type[ext])
                    self.end_headers()

                    with open(path) as ifp:
                        self.wfile.write(ifp.read())
                else:
                    # Unknown file type or a directory.
                    # Treat it as plain text.
                    self.send_response(200)  # OK
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()

                    with open(path) as ifp:
                        self.wfile.write(ifp.read())
            else:
                if dirpath is None:
                    # Invalid file path, respond with a server access error
                    self.send_response(500)  # generic server error for now
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    self.wfile.write('<html>')
                    self.wfile.write('  <head>')
                    self.wfile.write('    <title>Server Access Error</title>')
                    self.wfile.write('  </head>')
                    self.wfile.write('  <body>')
                    self.wfile.write('    <p>Server access error.</p>')
                    self.wfile.write('    <p>%r</p>' % (repr(self.path)))
                    self.wfile.write('    <p><a href="%s">Back</a></p>' % (rpath))
                    self.wfile.write('  </body>')
                    self.wfile.write('</html>')
                else:
                    # List the directory contents. Allow simple navigation.
                    logging.debug('DIR %s' % (dirpath))

                    self.send_response(200)  # OK
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    self.wfile.write('<html>')
                    self.wfile.write('  <head>')
                    self.wfile.write('    <title>%s</title>' % (dirpath))
                    self.wfile.write('  </head>')
                    self.wfile.write('  <body>')
                    self.wfile.write('    <a href="%s">Home</a><br>' % ('/'));

                    # Make the directory path navigable.
                    dirstr = ''
                    href = None
                    for seg in rpath.split('/'):
                        if href is None:
                            href = seg
                        else:
                            href = href + '/' + seg
                            dirstr += '/'
                        dirstr += '<a href="%s">%s</a>' % (href, seg)
                    self.wfile.write('    <p>Directory: %s</p>' % (dirstr))

                    # Write out the simple directory list (name and size).
                    self.wfile.write('    <table border="0">')
                    self.wfile.write('      <tbody>')
                    fnames = ['..']
                    fnames.extend(sorted(os.listdir(dirpath), key=str.lower))
                    for fname in fnames:
                        self.wfile.write('        <tr>')
                        self.wfile.write('          <td align="left">')
                        path = rpath + '/' + fname
                        fpath = os.path.join(dirpath, fname)
                        if os.path.isdir(path):
                            self.wfile.write('            <a href="%s">%s/</a>' % (path, fname))
                        else:
                            self.wfile.write('            <a href="%s">%s</a>' % (path, fname))
                        self.wfile.write('          <td>  </td>')
                        self.wfile.write('          </td>')
                        self.wfile.write('          <td align="right">%d</td>' % (os.path.getsize(fpath)))
                        self.wfile.write('        </tr>')
                    self.wfile.write('      </tbody>')
                    self.wfile.write('    </table>')
                    self.wfile.write('  </body>')
                    self.wfile.write('</html>')



    # def do_GET(s):
    #    s.send_response(200)
    #    s.send_header("Content-type", "text/html")
    #    s.end_headers()
    #
    #    f = open("..\wgo.player-2.0\go.html")
    #
    #    s.wfile.write(f.read())
    #    f.close()
    #
    #    if url[0] == "..\wgo.player-2.0\wgo\wgo.min.js":
    #        f = open("..\wgo.player-2.0\wgo\wgo.min.js", "rb")
    #
    #        for each_line in f:
    #            s.wfile.write(each_line)
    #        return

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)