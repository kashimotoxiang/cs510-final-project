# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import query_pb2 as query__pb2


class SentQueryStub(object):
  """The greeting service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ReturnResult = channel.unary_unary(
        '/query.SentQuery/ReturnResult',
        request_serializer=query__pb2.Query.SerializeToString,
        response_deserializer=query__pb2.Reply.FromString,
        )


class SentQueryServicer(object):
  """The greeting service definition.
  """

  def ReturnResult(self, request, context):
    """Sends a greeting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SentQueryServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ReturnResult': grpc.unary_unary_rpc_method_handler(
          servicer.ReturnResult,
          request_deserializer=query__pb2.Query.FromString,
          response_serializer=query__pb2.Reply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'query.SentQuery', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))