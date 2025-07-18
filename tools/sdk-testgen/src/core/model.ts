import 'reflect-metadata';
import * as _ from 'lodash';
import * as fs from 'fs';
import * as path from 'path';
import {
    ArraySchema,
    BinarySchema,
    BooleanSchema,
    CodeModel,
    ComplexSchema,
    DateTimeSchema,
    DictionarySchema,
    DurationSchema,
    ImplementationLocation,
    Languages,
    NumberSchema,
    ObjectSchema,
    Operation,
    OperationGroup,
    Parameter,
    Property,
    Schema,
    SchemaResponse,
    SchemaType,
    SecurityScheme,
    StringSchema,
    UriSchema,
    codeModelSchema,
} from '@autorest/codemodel';
import { AutorestExtensionHost, Session, startSession } from '@autorest/extension-base';
import { Config, OavStepType, testScenarioVariableDefault } from '../common/constant';
import { Helper } from '../util/helper';
import { TestConfig } from '../common/testConfig';

export enum ExtensionName {
    xMsExamples = 'x-ms-examples',
}
export interface ExampleExtensionResponse {
    body?: any;
    headers?: Record<string, any>;
}
export interface ExampleExtension {
    parameters?: Record<string, any>;
    responses?: Record<string, ExampleExtensionResponse>;
    // eslint-disable-next-line
    'x-ms-original-file'?: string;
}

/**
 * Generally a test group should be generated into one test source file.
 */
export class MockTestDefinitionModel {
    exampleGroups: ExampleGroup[] = [];
    public static groupByOperationGroup(exampleGroups: ExampleGroup[]): Record<string, ExampleGroup[]> {
        return exampleGroups.reduce((r, exampleGroup) => {
            const groupKey = exampleGroup.operationGroup.$key;
            r[groupKey] = r[groupKey] || [];
            r[groupKey].push(exampleGroup);
            return r;
        }, Object.create(null));
    }
}

export class ExampleGroup {
    operationId: string;
    operationGroup: OperationGroup;
    operation: Operation;
    examples: ExampleModel[] = [];
    public constructor(operationGroup: OperationGroup, operation: Operation, operationId: string) {
        this.operationGroup = operationGroup;
        this.operation = operation;
        this.operationId = operationId;
    }
}

export class TestModel {
    mockTest: MockTestDefinitionModel = new MockTestDefinitionModel();
}

export interface TestCodeModel extends CodeModel {
    testModel?: TestModel;
}

export class ExampleValue {
    language: Languages;
    schema: Schema;
    flattenedNames?: string[];

    /**Use elements if schema.type==Array, use properties if schema.type==Object/Dictionary, otherwise use rawValue */
    rawValue?: any;
    elements?: ExampleValue[];
    properties?: Record<string, ExampleValue>;

    parentsValue?: Record<string, ExampleValue>; // parent class Name--> value

    public constructor(schema: Schema = undefined, language: Languages = undefined) {
        this.language = language;
        this.schema = schema;
    }

    public get isNull(): boolean {
        return (
            (this.rawValue === null || this.rawValue === undefined) &&
            (this.elements === null || this.elements === undefined) &&
            (this.properties === null || this.properties === undefined)
        );
    }

    public static createInstance(
        session: Session<TestCodeModel>,
        rawValue: any,
        usedProperties: Set<string[]>,
        schema: Schema,
        language: Languages,
        extensions: Record<string, any>,
        searchDescents = true,
    ): ExampleValue {
        const xMsFormat = 'x-ms-format';
        const xMsFormatElementType = 'x-ms-format-element-type';
        const instance = new ExampleValue(schema, language);
        if (!schema) {
            instance.rawValue = rawValue;
            return instance;
        }

        function addParentValue(parent: ComplexSchema) {
            const parentValue = ExampleValue.createInstance(session, rawValue, usedProperties, parent, parent.language, parent.extensions, false);
            if ((parentValue.properties && Object.keys(parentValue.properties).length > 0) || (parentValue.parentsValue && Object.keys(parentValue.parentsValue).length > 0)) {
                instance.parentsValue[parent.language.default.name] = parentValue;
            }
        }

        if (schema.type === SchemaType.Array && Array.isArray(rawValue)) {
            instance.elements = rawValue.map((x) => this.createInstance(session, x, new Set(), (schema as ArraySchema).elementType, undefined, undefined));
        } else if (schema.type === SchemaType.Object && rawValue === Object(rawValue)) {
            const childSchema: ComplexSchema = searchDescents ? Helper.findInDescents(schema as ObjectSchema, rawValue) : schema;
            instance.schema = childSchema;

            instance.properties = {};
            const splitParentsValue = TestCodeModeler.instance.testConfig.getValue(Config.splitParentsValue);
            for (const property of Helper.getAllProperties(childSchema, !splitParentsValue)) {
                if (property.flattenedNames) {
                    if (!Helper.pathIsIncluded(usedProperties, property.flattenedNames)) {
                        const value = Helper.queryByPath(rawValue, property.flattenedNames);
                        if (value.length === 1) {
                            instance.properties[property.serializedName] = this.createInstance(
                                session,
                                value[0],
                                Helper.filterPathsByPrefix(usedProperties, property.flattenedNames),
                                property.schema,
                                property.language,
                                property.extensions,
                            );
                            instance.properties[property.serializedName].flattenedNames = property.flattenedNames;
                            usedProperties.add(property.flattenedNames);
                        }
                    }
                } else {
                    if (Object.prototype.hasOwnProperty.call(rawValue, property.serializedName) && !Helper.pathIsIncluded(usedProperties, [property.serializedName])) {
                        instance.properties[property.serializedName] = this.createInstance(
                            session,
                            rawValue[property.serializedName],
                            Helper.filterPathsByPrefix(usedProperties, [property.serializedName]),
                            property.schema,
                            property.language,
                            property.extensions,
                        );
                        usedProperties.add([property.serializedName]);
                    }
                }
            }

            instance.parentsValue = {};
            // Add normal parentValues
            if (splitParentsValue && Object.prototype.hasOwnProperty.call(childSchema, 'parents') && (childSchema as ObjectSchema).parents) {
                for (const parent of (childSchema as ObjectSchema).parents.immediate) {
                    if (childSchema.type === SchemaType.Object) {
                        addParentValue(parent);
                    } else {
                        console.warn(`${parent.language.default.name} is NOT a object type of parent of ${childSchema.language.default.name}!`);
                    }
                }
            }

            // Add AdditionalProperties as ParentValue
            if (Object.prototype.hasOwnProperty.call(childSchema, 'parents') && (childSchema as ObjectSchema).parents) {
                for (const parent of (childSchema as ObjectSchema).parents.immediate) {
                    if (parent.type === SchemaType.Dictionary) {
                        addParentValue(parent);
                    }
                }
            }
        } else if (schema.type === SchemaType.Dictionary && rawValue === Object(rawValue)) {
            instance.properties = {};
            for (const [key, value] of Object.entries(rawValue)) {
                if (!Helper.pathIsIncluded(usedProperties, [key])) {
                    instance.properties[key] = this.createInstance(session, value, new Set([...usedProperties]), (schema as DictionarySchema).elementType, undefined, undefined);
                    usedProperties.add([key]);
                }
            }
        } else if (schema.type === SchemaType.AnyObject && extensions && extensions[xMsFormat] && extensions[xMsFormat].startsWith('dfe-')) {
            // Becuase DataFactoryElement is defined as AnyObject schema, so have to explicitly build it's example value according to x_ms_format here
            const format = extensions[xMsFormat];
            const elementFormat = extensions[xMsFormatElementType];

            const dfeObjSchema = ExampleValue.createSchemaForDfeObject(session, rawValue, format);
            if (dfeObjSchema) {
                return this.createInstance(session, rawValue, usedProperties, dfeObjSchema, language, undefined, searchDescents);
            } else {
                const dfeLiterlSchema = ExampleValue.createSchemaForDfeLiteral(session, rawValue, format, elementFormat);
                return this.createInstance(session, rawValue, usedProperties, dfeLiterlSchema, language, undefined, searchDescents);
            }
        } else {
            instance.rawValue = rawValue;
        }
        return instance;
    }

    private static createSchemaForDfeObject(session: Session<TestCodeModel>, raw: any, dfeFormat: string): ObjectSchema | undefined {
        const dfeObjectType = 'type';
        const dfeObjectValue = 'value';
        const dfeObjectSchemaPrefix = 'DataFactoryElement-';

        if (Object(raw) && raw[dfeObjectType] && raw[dfeObjectValue]) {
            const r = new ObjectSchema(dfeObjectSchemaPrefix + raw[dfeObjectType], '');
            r.addProperty(new Property(dfeObjectType, '', new StringSchema(`${dfeFormat}-${dfeObjectType}`, '')));
            switch (raw[dfeObjectType]) {
                case 'Expression':
                case 'SecureString':
                    r.addProperty(new Property(dfeObjectValue, '', new StringSchema(`${dfeFormat}-${dfeObjectValue}`, '')));
                    return r;
                case 'AzureKeyVaultSecretReference': {
                    const valueSchema = session.model.schemas.objects.find((s) => s.language.default.name === `AzureKeyVaultSecretReference`);
                    if (!valueSchema) {
                        throw new Error('Cant find schema for the value of DataFactoryElement KeyVaultSecret Reference');
                    }
                    r.addProperty(new Property(dfeObjectValue, '', valueSchema));
                    return r;
                }
                default:
                    return undefined;
            }
        }
        return undefined;
    }

    private static createSchemaForDfeLiteral(session: Session<TestCodeModel>, raw: any, dfeFormat: string, eleFormat: string): Schema {
        switch (dfeFormat) {
            case 'dfe-string':
                return new StringSchema(dfeFormat, '');
            case 'dfe-bool':
                return new BooleanSchema(dfeFormat, '');
            case 'dfe-int':
                return new NumberSchema(dfeFormat, '', SchemaType.Integer, 32);
            case 'dfe-double':
                return new NumberSchema(dfeFormat, '', SchemaType.Number, 64);
            case 'dfe-date-time':
                return new DateTimeSchema(dfeFormat, '');
            case 'dfe-duration':
                return new DurationSchema(dfeFormat, '');
            case 'dfe-uri':
                return new UriSchema(dfeFormat, '');
            case 'dfe-list-string':
                return new ArraySchema(dfeFormat, '', new StringSchema(dfeFormat + '-element', ''));
            case 'dfe-key-value-pairs':
                return new DictionarySchema(dfeFormat, '', new StringSchema(dfeFormat + '-element', ''));
            case 'dfe-object':
                return new BinarySchema('');
            case 'dfe-list-generic': {
                // TODO: do we need to search more schema store for the element?
                // just searching object schemas seems enough for now. Consider add more when needed
                const eleSchema = session.model.schemas.objects.find((s) => s.language.default.name === eleFormat);
                if (!eleSchema) {
                    throw new Error('Cant find schema for the element of DataFactoryElement with type dfe-list-generic: ' + (eleFormat ?? '<null>'));
                }
                return new ArraySchema(dfeFormat, '', eleSchema);
            }
            default:
                throw new Error('Unknown dfeFormat' + dfeFormat);
        }
    }
}

export class ExampleParameter {
    /** Ref to the Parameter of operations in codeModel */
    parameter: Parameter;
    exampleValue: ExampleValue;

    public constructor(session: Session<TestCodeModel>, parameter: Parameter, rawValue: any) {
        this.parameter = parameter;
        this.exampleValue = ExampleValue.createInstance(session, rawValue, new Set(), parameter?.schema, parameter.language, parameter.extensions);
    }
}

export class SecurityParameter {
    schema: SecurityScheme; // Ref to security.schmas[x]
    rawValue: any;

    public constructor(schema: SecurityScheme, rawValue: any) {
        this.schema = schema;
        this.rawValue = rawValue;
    }
}

export class ExampleResponse {
    body?: ExampleValue;
    headers?: Record<string, any>;

    public static createInstance(session: Session<TestCodeModel>, rawResponse: ExampleExtensionResponse, schema: Schema, language: Languages): ExampleResponse {
        const instance = new ExampleResponse();
        if (rawResponse.body !== undefined) {
            instance.body = ExampleValue.createInstance(session, rawResponse.body, new Set(), schema, language, undefined);
        }
        instance.headers = rawResponse.headers;
        return instance;
    }
}

export class ExampleModel {
    /** Key in x-ms-examples */
    name: string;
    operationGroup: OperationGroup;
    operation: Operation;
    clientParameters: ExampleParameter[] = [];
    methodParameters: ExampleParameter[] = [];
    securityParameters?: SecurityParameter[] = undefined;
    responses: Record<string, ExampleResponse> = {}; // statusCode-->ExampleResponse
    originalFile: string;

    public constructor(name: string, operation: Operation, operationGroup: OperationGroup) {
        this.name = name;
        this.operation = operation;
        this.operationGroup = operationGroup;
    }
}

export class OutputVariableModel {
    index?: number;
    key?: string;
    languages?: Languages;
    public constructor(public type: OutputVariableModelType, value: number | string | Languages) {
        if (typeof value === 'number') {
            this.index = value;
        } else if (typeof value === 'string') {
            this.key = value;
        } else {
            this.languages = value;
        }
    }
}

export enum OutputVariableModelType {
    index = 'index',
    key = 'key',
    object = 'object',
}

function findResponseSchema(operation: Operation, statusCode: string): SchemaResponse {
    for (const response of operation.responses || []) {
        if ((response.protocol.http?.statusCodes || []).indexOf(statusCode) >= 0) {
            return response as SchemaResponse;
        }
    }
    return undefined;
}

export class TestCodeModeler {
    public static instance: TestCodeModeler;
    public testConfig: TestConfig;
    private constructor(public codeModel: TestCodeModel, testConfig: TestConfig) {
        this.testConfig = testConfig;
    }

    public static createInstance(codeModel: TestCodeModel, testConfig: TestConfig): TestCodeModeler {
        if (TestCodeModeler.instance) {
            TestCodeModeler.instance.codeModel = codeModel;
            TestCodeModeler.instance.testConfig = testConfig;
        }
        TestCodeModeler.instance = new TestCodeModeler(codeModel, testConfig);
        return TestCodeModeler.instance;
    }

    private isBodyParameter(parameter: Parameter): boolean {
        return (
            parameter.protocol?.http?.in === 'body' ||
            (Object.prototype.hasOwnProperty.call(parameter, 'originalParameter') && this.isBodyParameter(parameter['originalParameter'] as Parameter))
        );
    }

    private createExampleModel(
        session: Session<TestCodeModel>,
        exampleExtension: ExampleExtension,
        exampleName,
        operation: Operation,
        operationGroup: OperationGroup,
    ): ExampleModel {
        const allParameters = Helper.allParameters(operation);
        const parametersInExample = exampleExtension.parameters;
        const exampleModel = new ExampleModel(exampleName, operation, operationGroup);
        exampleModel.originalFile = Helper.getExampleRelativePath(exampleExtension['x-ms-original-file']);
        for (const parameter of allParameters) {
            if (parameter.flattened) {
                continue;
            }
            const t = Helper.getFlattenedNames(parameter);
            const paramRawData = this.isBodyParameter(parameter) ? Helper.queryBodyParameter(parametersInExample, t) : Helper.queryByPath(parametersInExample, t);
            if (paramRawData.length === 1) {
                const exampleParameter = new ExampleParameter(session, parameter, paramRawData[0]);
                if (parameter.implementation === ImplementationLocation.Method) {
                    exampleModel.methodParameters.push(exampleParameter);
                } else if (parameter.implementation === ImplementationLocation.Client) {
                    exampleModel.clientParameters.push(exampleParameter);
                } else {
                    // ignore
                }
            }
        }

        for (const paramName of Object.keys(parametersInExample)) {
            for (const securitySchema of this.codeModel.security.schemes) {
                if (Object.prototype.hasOwnProperty.call(securitySchema, 'name') && paramName === securitySchema['name']) {
                    if (exampleModel.securityParameters === undefined) {
                        exampleModel.securityParameters = [];
                    }
                    exampleModel.securityParameters.push(new SecurityParameter(securitySchema, parametersInExample[paramName]));
                    break;
                }
            }
        }

        for (const [statusCode, response] of Object.entries(exampleExtension.responses)) {
            const exampleExtensionResponse = response;
            const schemaResponse = findResponseSchema(operation, statusCode);
            if (schemaResponse) {
                exampleModel.responses[statusCode] = ExampleResponse.createInstance(session, exampleExtensionResponse, schemaResponse.schema, schemaResponse.language);
            }
        }
        return exampleModel;
    }

    public initiateTests() {
        if (!this.codeModel.testModel) {
            this.codeModel.testModel = new TestModel();
        }
    }

    public genMockTests(session: Session<TestCodeModel>) {
        this.initiateTests();
        if (!this.testConfig.getValue(Config.useExampleModel)) {
            return;
        }
        this.codeModel.operationGroups.forEach((operationGroup) => {
            operationGroup.operations.forEach((operation) => {
                const operationId = operation.operationId ? operation.operationId : operationGroup.language.default.name + '_' + operation.language.default.name;
                // TODO: skip non-json http bodys for now. Need to validate example type with body schema to support it.
                const mediaTypes = operation.requests[0]?.protocol?.http?.mediaTypes;
                if (mediaTypes && mediaTypes.indexOf('application/json') < 0) {
                    session.warning(`genMockTests: MediaTypes ${operation.requests[0]?.protocol?.http?.mediaTypes} in operation ${operationId} is not supported!`, [
                        'Test Modeler',
                    ]);
                    return;
                }
                const exampleGroup = new ExampleGroup(operationGroup, operation, operationId);
                for (const [exampleName, rawValue] of Object.entries(operation.extensions?.[ExtensionName.xMsExamples] ?? {})) {
                    if (!this.testConfig.isDisabledExample(exampleName)) {
                        exampleGroup.examples.push(this.createExampleModel(session, rawValue as ExampleExtension, exampleName, operation, operationGroup));
                    }
                }
                this.codeModel.testModel.mockTest.exampleGroups.push(exampleGroup);
            });
        });
    }

    public static async getSessionFromHost(host: AutorestExtensionHost) {
        return await startSession<TestCodeModel>(host, codeModelSchema);
    }

    public findOperationByOperationId(
        operationId: string,
    ): {
        operation: Operation;
        operationGroup: OperationGroup;
    } {
        let [groupKey, operationName] = operationId.split('_');
        if (!operationName) {
            operationName = groupKey;
            groupKey = '';
        }
        for (const group of this.codeModel.operationGroups) {
            if (group.$key !== groupKey) {
                continue;
            }
            for (const op of group.operations) {
                if (op.language.default.name === operationName) {
                    return {
                        operation: op,
                        operationGroup: group,
                    };
                }
            }
        }
        return undefined;
    }

    public createApiScenarioLoaderOption(fileRoot: string) {
        const options = {
            useJsonParser: false,
            checkUnderFileRoot: false,
            fileRoot: fileRoot,
            swaggerFilePaths: this.testConfig.getValue(Config.inputFile),
            eraseXmsExamples: false,
        };
        return { ...options, ...this.testConfig.getValue(Config.apiScenarioLoaderOption, {}) };
    }

}
